"""Legal Retrieval v2 QuerySpec package."""

from app.rag.legal_v2.query.query_spec import (
    ConstraintCategory,
    ConstraintPolarity,
    QueryConstraint,
    QueryDateRange,
    QueryEntity,
    QueryEvent,
    QueryIntent,
    QueryRelation,
    QuerySpecV2,
    build_query_spec_v2,
)

__all__ = [
    "ConstraintCategory",
    "ConstraintPolarity",
    "QueryConstraint",
    "QueryDateRange",
    "QueryEntity",
    "QueryEvent",
    "QueryIntent",
    "QueryRelation",
    "QuerySpecV2",
    "build_query_spec_v2",
]
