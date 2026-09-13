use std::sync::{Arc, LazyLock, OnceLock};

use arrow_schema::{DataType, FieldRef};
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::scalar_doc_sections::DOC_SECTION_OTHER;
use datafusion::logical_expr::{
    ColumnarValue, Documentation, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl, Signature,
};
use geo_traits::{GeometryTrait, PolygonTrait};
use geoarrow_array::array::{GeometryArray, PolygonArray, from_arrow_array};
use geoarrow_array::builder::GeometryBuilder;
use geoarrow_array::cast::AsGeoArrowArray;
use geoarrow_array::{GeoArrowArray, GeoArrowArrayAccessor, downcast_geoarrow_array};
use geoarrow_schema::{CoordType, GeoArrowType, GeometryType, Metadata};

use crate::data_types::any_single_geometry_type_input;
use crate::error::GeoDataFusionResult;

#[derive(Debug, Eq, PartialEq, Hash)]
pub struct ExteriorRing {
    coord_type: CoordType,
}

impl ExteriorRing {
    pub fn new(coord_type: CoordType) -> Self {
        Self { coord_type }
    }
}

impl Default for ExteriorRing {
    fn default() -> Self {
        Self::new(Default::default())
    }
}

static DOCUMENTATION: OnceLock<Documentation> = OnceLock::new();
static ALIASES: LazyLock<Vec<String>> = LazyLock::new(|| vec!["st_exteriorring".to_string()]);

impl ScalarUDFImpl for ExteriorRing {
    fn name(&self) -> &str {
        "st_exteriorring"
    }

    fn signature(&self) -> &Signature {
        any_single_geometry_type_input()
    }

    fn aliases(&self) -> &[String] {
        &ALIASES
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Err(DataFusionError::Internal(
            "Return field is computed from metadata in return_field_from_args.".to_string(),
        ))
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let metadata =
            Arc::new(Metadata::try_from(args.arg_fields[0].as_ref()).unwrap_or_default());
        let output_type = GeometryType::new(metadata).with_coord_type(self.coord_type);
        Ok(Arc::new(output_type.to_field("", true)))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        Ok(exterior_ring_impl(args, self.coord_type)?)
    }

    fn documentation(&self) -> Option<&Documentation> {
        Some(DOCUMENTATION.get_or_init(|| {
            Documentation::builder(
                DOC_SECTION_OTHER,
                "Return the exterior ring of a polygon geometry.",
                "ST_ExteriorRing(geometry)",
            )
            .with_argument("g1", "geometry")
            .build()
        }))
    }
}

fn exterior_ring_impl(
    args: ScalarFunctionArgs,
    coord_type: CoordType,
) -> GeoDataFusionResult<ColumnarValue> {
    let arrays = ColumnarValue::values_to_arrays(&args.args)?;
    let geo_array = from_arrow_array(&arrays[0], &args.arg_fields[0])?;
    let geom_type =
        GeometryType::new(geo_array.data_type().metadata().clone()).with_coord_type(coord_type);

    let out: GeometryArray = match geo_array.data_type() {
        GeoArrowType::Polygon(_) => polygon_impl(geo_array.as_polygon(), &geom_type)?,
        _ => {
            let geo_array_ref = geo_array.as_ref();
            downcast_geoarrow_array!(geo_array_ref, geometry_impl, &geom_type)?
        }
    };

    Ok(out.into_array_ref().into())
}

fn polygon_impl(
    array: &PolygonArray,
    geom_type: &GeometryType,
) -> GeoDataFusionResult<GeometryArray> {
    let mut builder = GeometryBuilder::new(geom_type.clone());

    for item in array.iter() {
        if let Some(geom) = item {
            match geom?.exterior() {
                Some(ring) => builder.push_geometry(Some(&ring))?,
                None => builder.push_null(),
            }
        } else {
            builder.push_null();
        }
    }

    Ok(builder.finish())
}

fn geometry_impl<'a>(
    array: &'a impl GeoArrowArrayAccessor<'a>,
    geom_type: &GeometryType,
) -> GeoDataFusionResult<GeometryArray> {
    let mut builder = GeometryBuilder::new(geom_type.clone());

    for item in array.iter() {
        if let Some(geom) = item {
            match geom?.as_type() {
                geo_traits::GeometryType::Polygon(polygon) => match polygon.exterior() {
                    Some(ring) => builder.push_geometry(Some(&ring))?,
                    None => builder.push_null(),
                },
                _ => builder.push_null(),
            }
        } else {
            builder.push_null();
        }
    }

    Ok(builder.finish())
}

#[cfg(test)]
mod test {
    use arrow_array::cast::AsArray;
    use datafusion::prelude::SessionContext;
    use geoarrow_array::array::from_arrow_array;
    use geoarrow_array::cast::to_wkt;

    use super::*;
    use crate::udf::native::io::GeomFromText;

    fn ctx() -> SessionContext {
        let ctx = SessionContext::new();
        ctx.register_udf(ExteriorRing::default().into());
        ctx.register_udf(GeomFromText::new(Default::default()).into());
        ctx
    }

    async fn exterior_ring_wkt(ctx: &SessionContext, wkt: &str) -> Option<String> {
        let df = ctx
            .sql(&format!("SELECT ST_ExteriorRing(ST_GeomFromText('{wkt}'))"))
            .await
            .unwrap();
        let batch = df.collect().await.unwrap().into_iter().next().unwrap();
        let field = batch.schema().field(0).clone();
        let geo = from_arrow_array(batch.column(0).as_ref(), &field).unwrap();
        if geo.as_ref().is_null(0) {
            return None;
        }
        let wkt_array = to_wkt::<i32>(geo.as_ref()).unwrap().into_array_ref();
        Some(wkt_array.as_string::<i32>().value(0).to_string())
    }

    #[tokio::test]
    async fn test_exterior_ring_of_polygon_with_hole() {
        let ctx = ctx();
        let wkt = exterior_ring_wkt(&ctx, "POLYGON((0 0,0 3,3 3,3 0,0 0),(1 1,1 2,2 2,2 1,1 1))")
            .await
            .unwrap();
        assert_eq!(wkt, "LINESTRING(0 0,0 3,3 3,3 0,0 0)");
    }

    #[tokio::test]
    async fn test_exterior_ring_of_non_polygon_is_null() {
        let ctx = ctx();
        assert_eq!(exterior_ring_wkt(&ctx, "POINT(1 2)").await, None);
        assert_eq!(exterior_ring_wkt(&ctx, "LINESTRING(0 0,1 1)").await, None);
    }
}
