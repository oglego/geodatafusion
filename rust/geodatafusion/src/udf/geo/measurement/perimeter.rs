use std::sync::{Arc, LazyLock, OnceLock};

use arrow_array::Float64Array;
use arrow_array::builder::Float64Builder;
use arrow_schema::DataType;
use datafusion::error::Result;
use datafusion::logical_expr::scalar_doc_sections::DOC_SECTION_OTHER;
use datafusion::logical_expr::{
    ColumnarValue, Documentation, ScalarFunctionArgs, ScalarUDFImpl, Signature,
};
use geo::{Euclidean, Length};
use geo_traits::GeometryTrait;
use geo_traits::to_geo::{ToGeoMultiPolygon, ToGeoPolygon};
use geoarrow_array::array::from_arrow_array;
use geoarrow_array::{GeoArrowArrayAccessor, downcast_geoarrow_array};

use crate::data_types::any_single_geometry_type_input;
use crate::error::GeoDataFusionResult;

/// `ST_Perimeter` scalar UDF implementation: returns the 2D Cartesian length of the boundary of
/// polygonal geometries.
#[derive(Debug, Eq, PartialEq, Hash)]
pub struct Perimeter;

impl Perimeter {
    /// Creates a new `ST_Perimeter` UDF instance.
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for Perimeter {
    fn default() -> Self {
        Self::new()
    }
}

static ALIASES: LazyLock<Vec<String>> = LazyLock::new(|| vec!["st_perimeter2d".to_string()]);
static DOCUMENTATION: OnceLock<Documentation> = OnceLock::new();

impl ScalarUDFImpl for Perimeter {
    fn name(&self) -> &str {
        "st_perimeter"
    }

    fn aliases(&self) -> &[String] {
        ALIASES.as_slice()
    }

    fn signature(&self) -> &Signature {
        any_single_geometry_type_input()
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Float64)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        Ok(perimeter_impl(args)?)
    }

    fn documentation(&self) -> Option<&Documentation> {
        Some(DOCUMENTATION.get_or_init(|| {
            Documentation::builder(
                DOC_SECTION_OTHER,
                "Returns the 2D Cartesian length of the boundary (exterior and interior rings) of a Polygon or MultiPolygon. For non-polygonal geometries 0 is returned; use ST_Length for linear geometries.",
                "ST_Perimeter(geom)",
            )
            .with_argument("geom", "geometry")
            .with_sql_example(
                "SELECT ST_Perimeter(ST_GeomFromText('POLYGON((0 0, 4 0, 4 3, 0 3, 0 0))'));",
            )
            .build()
        }))
    }
}

fn perimeter_impl(args: ScalarFunctionArgs) -> GeoDataFusionResult<ColumnarValue> {
    let array = ColumnarValue::values_to_arrays(&args.args)?
        .into_iter()
        .next()
        .unwrap();
    let geo_array = from_arrow_array(&array, &args.arg_fields[0])?;
    let geo_array_ref = geo_array.as_ref();
    let result = downcast_geoarrow_array!(geo_array_ref, perimeter_array)?;
    Ok(ColumnarValue::Array(Arc::new(result)))
}

fn perimeter_array<'a>(
    array: &'a impl GeoArrowArrayAccessor<'a>,
) -> GeoDataFusionResult<Float64Array> {
    let mut builder = Float64Builder::with_capacity(array.len());
    for geometry in array.iter() {
        match geometry {
            Some(geometry) => {
                let perimeter = match geometry?.as_type() {
                    geo_traits::GeometryType::Polygon(polygon) => {
                        polygon_perimeter(&polygon.to_polygon())
                    }
                    geo_traits::GeometryType::MultiPolygon(multi_polygon) => multi_polygon
                        .to_multi_polygon()
                        .iter()
                        .map(polygon_perimeter)
                        .sum(),
                    _ => 0.0,
                };
                builder.append_value(perimeter);
            }
            None => builder.append_null(),
        }
    }
    Ok(builder.finish())
}

/// Total length of a polygon's exterior ring plus all of its interior rings.
fn polygon_perimeter(polygon: &geo::Polygon<f64>) -> f64 {
    Euclidean.length(polygon.exterior())
        + polygon
            .interiors()
            .iter()
            .map(|ring| Euclidean.length(ring))
            .sum::<f64>()
}

#[cfg(test)]
mod test {
    use arrow_array::Array;
    use arrow_array::cast::AsArray;
    use arrow_array::types::Float64Type;
    use datafusion::prelude::SessionContext;

    use super::*;
    use crate::udf::native::io::GeomFromText;

    async fn perimeter(sql: &str) -> Option<f64> {
        let ctx = SessionContext::new();
        ctx.register_udf(Perimeter::new().into());
        ctx.register_udf(GeomFromText::new(Default::default()).into());

        let batch = ctx
            .sql(sql)
            .await
            .unwrap()
            .collect()
            .await
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let col = batch.column(0).as_primitive::<Float64Type>();
        (!col.is_null(0)).then(|| col.value(0))
    }

    #[tokio::test]
    async fn test_polygon_perimeter() {
        let val =
            perimeter("SELECT ST_Perimeter(ST_GeomFromText('POLYGON((0 0, 4 0, 4 3, 0 3, 0 0))'))")
                .await;
        assert_eq!(val, Some(14.0));
    }

    #[tokio::test]
    async fn test_polygon_with_hole_includes_interior_ring() {
        let val = perimeter(
            "SELECT ST_Perimeter(ST_GeomFromText('POLYGON((0 0, 10 0, 10 10, 0 10, 0 0), (2 2, 4 2, 4 4, 2 4, 2 2))'))",
        )
        .await;
        // 40 for the exterior ring plus 8 for the hole.
        assert_eq!(val, Some(48.0));
    }

    #[tokio::test]
    async fn test_multipolygon_perimeter() {
        let val = perimeter(
            "SELECT ST_Perimeter(ST_GeomFromText('MULTIPOLYGON(((0 0, 1 0, 1 1, 0 1, 0 0)), ((10 10, 12 10, 12 12, 10 12, 10 10)))'))",
        )
        .await;
        assert_eq!(val, Some(12.0));
    }

    #[tokio::test]
    async fn test_non_polygonal_is_zero() {
        let val = perimeter("SELECT ST_Perimeter(ST_GeomFromText('LINESTRING(0 0, 3 4)'))").await;
        assert_eq!(val, Some(0.0));
    }

    #[tokio::test]
    async fn test_null_is_null() {
        let val = perimeter("SELECT ST_Perimeter(ST_GeomFromText(CAST(NULL AS VARCHAR)))").await;
        assert_eq!(val, None);
    }
}
