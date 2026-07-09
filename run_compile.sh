vai_c_xir \
  --xmodel quantize_result_customnet5_cont_512_512/Novelty_ULite_int.xmodel \
  --arch quantized/arch.json \
  --output_dir compiled \
  --net_name semantic_ulite_custom5_cont_kr260

echo "-----------------------------------------"
echo "MODEL COMPILED"
echo "-----------------------------------------"

