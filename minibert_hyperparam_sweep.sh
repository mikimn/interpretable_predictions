#lambda1_candidates=(0.01 0.05 0.1 0.15)
lambda1_candidates=(0.01 0.05 0.1)
#lambda2_candidates=(0.001 0.005 0.01 0.015 0.02)
lambda2_candidates=(0.001 0.005 0.01)
seed=42
model="google/bert_uncased_L-4_H-256_A-4"

for lambda1 in "${lambda1_candidates[@]}"; do
  for lambda2 in "${lambda2_candidates[@]}"; do
    for mask_before_layer in {0..2}; do
      ./py-sbatch.sh esnli.py \
        --model_name_or_path "$model" \
        --rationale_type all \
        --tag "minibert-mask-all-sparsity${lambda1}-lasso${lambda2}-layer${mask_before_layer}" \
        --lambda_init "${lambda1}" \
        --lambda_lasso "${lambda2}" \
        --seed "${seed}" \
        --mask_before_layer "${mask_before_layer}"
    done
  done
done
