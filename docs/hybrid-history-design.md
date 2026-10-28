# Superseded: Hybrid Decision History

This design has been superseded by [Append-only Conversation History and Compaction](conversation-history-design.md).

The `hybrid_v1` policy and bounded decision journal remain documented here only
so older artifacts and implementation history remain understandable. New
interactive work must use `conversation_v1`, which preserves the exact ordered
Mistral conversation and compacts it at a 500,000-token application threshold.
