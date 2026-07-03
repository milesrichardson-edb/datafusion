// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! End-to-end regression tests for a `UNION ALL` field-metadata mismatch
//! between the logical and physical planner.
//!
//! `datafusion/expr/src/expr.rs::intersect_metadata_for_union` is the
//! canonical way DataFusion combines a union field's per-branch metadata:
//! keys are kept only if every *non-empty* branch agrees on the value,
//! otherwise the key is dropped. `Union::try_new` uses it. However, until
//! this fix, two other call sites computed the same information with
//! different ("merge", last-writer-wins) semantics instead of calling the
//! shared helper:
//!
//!   - `datafusion/optimizer/src/analyzer/type_coercion.rs::coerce_union_schema_with_schema`
//!     (the schema the `TypeCoercion` analyzer rebuilds *every*
//!     `LogicalPlan::Union` with -- the logical schema normally compared
//!     during physical planning), and
//!   - `datafusion/physical-plan/src/union.rs::union_schema` (shared by
//!     `UnionExec` and `InterleaveExec`).
//!
//! Because the two merge sites pick their "last writer" differently
//! (physical: `find_or_first(Field::is_nullable)`, i.e. nullability-driven;
//! logical: plain sequential input order), they can disagree with each
//! other even though neither one is the intersect path. When they disagree,
//! `datafusion/core/src/physical_planner.rs`'s `LogicalPlan::Aggregate` arm
//! trips its exact-schema-equality check (`schema_satisfied_by`) and
//! returns `Internal error: Physical input schema should be the same as
//! the one converted from logical input schema. Differences: ...`.
//!
//! The first test below reproduces this with a `UNION ALL` between a
//! non-nullable field and a nullable field that both carry a conflicting
//! `PARQUET:field_id`-shaped metadata key. The asymmetric nullability is
//! required: physical `union_schema`'s `find_or_first(Field::is_nullable)`
//! prefers the nullable branch, while the logical merge picks whichever
//! branch was written last in plain input order -- with both branches
//! nullable (or non-nullable), the two "last writer" choices happen to
//! agree, which masks the bug entirely. See `ADVERSARIAL-REVIEW-FINDINGS.md`
//! Finding #3 in the accompanying RCA materials.
//!
//! The second test is a same-shape regression guard for the *already
//! fixed* one-branch-empty-metadata case (the union.rs/type_coercion.rs
//! merge sites never dropped metadata on an empty branch, so this passed
//! before this fix too -- it must keep passing now that both sites call
//! `intersect_metadata_for_union`, since that skips empty maps rather than
//! treating them as "no metadata should survive").

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::Int64Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::assert_batches_eq;
use datafusion::catalog::MemTable;
use datafusion::prelude::*;
use datafusion_common::Result;

fn field_id_metadata(id: &str) -> HashMap<String, String> {
    HashMap::from([("PARQUET:field_id".to_string(), id.to_string())])
}

fn register_int64_column(
    ctx: &SessionContext,
    table_name: &str,
    column_name: &str,
    nullable: bool,
    metadata: HashMap<String, String>,
    values: Vec<Option<i64>>,
) -> Result<()> {
    let field =
        Field::new(column_name, DataType::Int64, nullable).with_metadata(metadata);
    let schema = Arc::new(Schema::new(vec![field]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from(values)) as _],
    )?;
    ctx.register_table(
        table_name,
        Arc::new(MemTable::try_new(schema, vec![vec![batch]])?),
    )?;
    Ok(())
}

/// Regression test: `UNION ALL` of two branches with ASYMMETRIC nullability
/// and CONFLICTING non-empty field metadata must still plan and execute.
///
/// Before this fix: fails during physical planning of the outer aggregate
/// with `Internal error: Physical input schema should be the same as the
/// one converted from logical input schema...` because the logical
/// (`coerce_union_schema_with_schema`) and physical (`union_schema`) merge
/// sites pick different "winning" branches for the conflicting
/// `PARQUET:field_id` key.
#[tokio::test]
async fn union_all_asymmetric_nullability_conflicting_field_metadata_aggregates()
-> Result<()> {
    let ctx = SessionContext::new();

    // t1.a: NOT NULL, field_id=1
    register_int64_column(
        &ctx,
        "t1",
        "a",
        false,
        field_id_metadata("1"),
        vec![Some(1), Some(2), Some(3)],
    )?;
    // t2.b: NULLABLE, field_id=2 -- conflicts with t1.a's field_id
    register_int64_column(
        &ctx,
        "t2",
        "b",
        true,
        field_id_metadata("2"),
        vec![Some(10), None, Some(20)],
    )?;

    let sql = "SELECT sum(v) AS total FROM \
               (SELECT a AS v FROM t1 UNION ALL SELECT b AS v FROM t2) u";

    // Planning must succeed (this is where the internal error was thrown).
    let df = ctx.sql(sql).await?;

    // Execution must succeed and produce the correct sum:
    // (1 + 2 + 3) + (10 + 20) = 36
    let batches = df.collect().await?;
    assert_batches_eq!(
        [
            "+-------+",
            "| total |",
            "+-------+",
            "| 36    |",
            "+-------+",
        ],
        &batches
    );

    Ok(())
}

/// Regression guard: one branch has EMPTY field metadata, the other has a
/// concrete value -- the non-empty value must survive (skip-empty
/// semantics from `intersect_metadata_for_union`, already correct via
/// `Union::try_new` per #21127; this guards that the two merge-turned-
/// intersect call sites (`union_schema`, `coerce_union_schema_with_schema`)
/// don't regress it into "any empty branch wipes the key").
#[tokio::test]
async fn union_all_field_metadata_empty_branch_skipped() -> Result<()> {
    let ctx = SessionContext::new();

    // t3.a: NOT NULL, field_id=1
    register_int64_column(
        &ctx,
        "t3",
        "a",
        false,
        field_id_metadata("1"),
        vec![Some(1), Some(2)],
    )?;
    // t4.b: NULLABLE, no metadata at all
    register_int64_column(&ctx, "t4", "b", true, HashMap::new(), vec![Some(5), None])?;

    let sql = "SELECT sum(v) AS total FROM \
               (SELECT a AS v FROM t3 UNION ALL SELECT b AS v FROM t4) u";

    let df = ctx.sql(sql).await?;
    let batches = df.collect().await?;
    assert_batches_eq!(
        [
            "+-------+",
            "| total |",
            "+-------+",
            "| 8     |",
            "+-------+",
        ],
        &batches
    );

    Ok(())
}
