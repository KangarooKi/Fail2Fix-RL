# Mimo Teacher Corrections for GSM8K Failure-to-Fix

This folder releases the teacher-correction dataset used for the teacher-guided F2F pilot experiments.

## Files

```text
mimo_teacher_corrections_gsm8k_n3676.jsonl   Verified correction SFT examples.
manifest.json                                Dataset statistics and schema summary.
```

## Dataset Summary

- **Examples**: 3,676 JSONL rows
- **Source problems**: GSM8K train split from ModelScope (`AI-ModelScope/gsm8k`)
- **Student model**: `Qwen/Qwen2.5-0.5B-Instruct`
- **Teacher model**: `mimo-v2.5-pro`
- **Use case**: correction SFT initialization before online F2F RL
- **Verification**: all released rows have `teacher_verified=true` and `teacher_format_ok=true`

## JSONL Schema

Each row contains both metadata and the SFT pair:

- `prompt`: correction prompt shown to the student during SFT
- `response`: verified teacher correction target
- `question`, `reference_answer`: original GSM8K problem and answer
- `student_solution`: failed or low-pass-rate student rollout
- `student_predicted_answer`: answer extracted from the student rollout
- `teacher_response`: raw teacher correction
- `teacher_error_type`, `teacher_error_location`, `teacher_error`: structured diagnosis
- `teacher_final_answer`, `teacher_predicted_answer`: extracted final teacher answer
- `pass_rate`: student rollout group pass rate for this problem
- `teacher_verified`, `teacher_format_ok`: filtering flags

For standard supervised fine-tuning, use only:

```json
{"prompt": "...", "response": "..."}
```

## Notes

This is a research artifact generated for Failure-to-Fix correction experiments. It includes GSM8K questions and generated teacher corrections, so users should respect the upstream GSM8K/ModelScope dataset terms and the terms of the teacher-model provider. No API keys or private endpoints are included.
