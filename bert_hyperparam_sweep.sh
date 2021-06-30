
lambda1_candidates=(0.05 0.08 0.1 0.15)
# lambda2_candidates=(0.001 0.005 0.01 0.015 0.02)
lambda2_candidates=(0.005 0.01 0.015 0.02)
seed=142
model="bert-base-uncased"

for lambda1 in "${lambda1_candidates[@]}"; do
  for lambda2 in "${lambda2_candidates[@]}"; do
    ./py-sbatch.sh esnli.py \
      --model_name_or_path "$model" \
      --rationale_type all \
      --tag "bert-mask-all-sparsity${lambda1}-lasso${lambda2}" \
      --lambda_init "${lambda1}" \
      --lambda_lasso "${lambda2}" \
      --seed "${seed}"
  done
done
