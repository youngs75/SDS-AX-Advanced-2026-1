## L40s 48G - Pytorch Jupyter notebook


```bash
pip install vllm
pip install huggingface_hub
```

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3.5-9B")
params = SamplingParams(temperature=0)
outputs = llm.generate("Hello, my name is", params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```


```log
INFO 04-07 02:43:24 [utils.py:233] non-default args: {'disable_log_stats': True, 'model': 'Qwen/Qwen3.5-9B'}
INFO 04-07 02:43:34 [model.py:549] Resolved architecture: Qwen3_5ForConditionalGeneration
INFO 04-07 02:43:34 [model.py:1678] Using max model len 262144
INFO 04-07 02:43:34 [scheduler.py:238] Chunked prefill is enabled with max_num_batched_tokens=8192.
INFO 04-07 02:43:34 [config.py:281] Setting attention block size to 528 tokens to ensure that attention page size is >= mamba page size.
INFO 04-07 02:43:34 [config.py:312] Padding mamba page size by 0.76% to ensure that mamba page size and attention page size are exactly equal.
INFO 04-07 02:43:34 [vllm.py:790] Asynchronous scheduling is enabled.
WARNING 04-07 02:43:41 [system_utils.py:152] We must use the `spawn` multiprocessing start method. Overriding VLLM_WORKER_MULTIPROC_METHOD to 'spawn'. See https://docs.vllm.ai/en/latest/usage/troubleshooting.html#python-multiprocessing for more information. Reasons: CUDA is initialized
(EngineCore pid=1462) INFO 04-07 02:43:51 [core.py:105] Initializing a V1 LLM engine (v0.19.0) with config: model='Qwen/Qwen3.5-9B', speculative_config=None, tokenizer='Qwen/Qwen3.5-9B', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=262144, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, decode_context_parallel_size=1, dcp_comm_backend=ag_rs, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, enable_return_routed_experts=False, kv_cache_dtype=auto, device_config=cuda, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser='', reasoning_parser_plugin='', enable_in_reasoning=False), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, kv_cache_metrics=False, kv_cache_metrics_sample=0.01, cudagraph_metrics=False, enable_layerwise_nvtx_tracing=False, enable_mfu_metrics=False, enable_mm_processor_stats=False, enable_logging_iteration_details=False), seed=0, served_model_name=Qwen/Qwen3.5-9B, enable_prefix_caching=False, enable_chunked_prefill=True, pooler_config=None, compilation_config={'mode': <CompilationMode.VLLM_COMPILE: 3>, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'inductor', 'custom_ops': ['none'], 'splitting_ops': ['vllm::unified_attention', 'vllm::unified_attention_with_output', 'vllm::unified_mla_attention', 'vllm::unified_mla_attention_with_output', 'vllm::mamba_mixer2', 'vllm::mamba_mixer', 'vllm::short_conv', 'vllm::linear_attention', 'vllm::plamo2_mamba_mixer', 'vllm::gdn_attention_core', 'vllm::olmo_hybrid_gdn_full_forward', 'vllm::kda_attention', 'vllm::sparse_attn_indexer', 'vllm::rocm_aiter_sparse_attn_indexer', 'vllm::unified_kv_cache_update', 'vllm::unified_mla_kv_cache_update'], 'compile_mm_encoder': False, 'cudagraph_mm_encoder': False, 'encoder_cudagraph_token_budgets': [], 'encoder_cudagraph_max_images_per_batch': 0, 'compile_sizes': [], 'compile_ranges_endpoints': [8192], 'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'size_asserts': False, 'alignment_asserts': False, 'scalar_asserts': False, 'combo_kernels': True, 'benchmark_combo_kernel': True}, 'inductor_passes': {}, 'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>, 'cudagraph_num_of_warmups': 1, 'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512], 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': False, 'pass_config': {'fuse_norm_quant': False, 'fuse_act_quant': False, 'fuse_attn_quant': False, 'enable_sp': False, 'fuse_gemm_comms': False, 'fuse_allreduce_rms': False}, 'max_cudagraph_capture_size': 512, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, 'assume_32_bit_indexing': False}, 'local_cache_dir': None, 'fast_moe_cold_start': True, 'static_all_moe_layers': []}
(EngineCore pid=1462) INFO 04-07 02:43:53 [parallel_state.py:1400] world_size=1 rank=0 local_rank=0 distributed_init_method=tcp://172.27.0.2:43181 backend=nccl
(EngineCore pid=1462) INFO 04-07 02:43:53 [parallel_state.py:1716] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, PCP rank 0, TP rank 0, EP rank N/A, EPLB rank N/A
(EngineCore pid=1462) INFO 04-07 02:43:58 [gpu_model_runner.py:4735] Starting to load model Qwen/Qwen3.5-9B...
(EngineCore pid=1462) INFO 04-07 02:43:59 [cuda.py:390] Using backend AttentionBackendEnum.FLASH_ATTN for vit attention
(EngineCore pid=1462) INFO 04-07 02:43:59 [mm_encoder_attention.py:230] Using AttentionBackendEnum.FLASH_ATTN for MMEncoderAttention.
(EngineCore pid=1462) INFO 04-07 02:43:59 [gdn_linear_attn.py:147] Using Triton/FLA GDN prefill kernel
(EngineCore pid=1462) INFO 04-07 02:43:59 [cuda.py:334] Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION'].
(EngineCore pid=1462) INFO 04-07 02:43:59 [flash_attn.py:596] Using FlashAttention version 2
(EngineCore pid=1462) <frozen importlib._bootstrap_external>:1297: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
(EngineCore pid=1462) <frozen importlib._bootstrap_external>:1297: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:07<00:21,  7.16s/it]
Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:14<00:14,  7.09s/it]
Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:21<00:07,  7.10s/it]
Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:25<00:00,  6.15s/it]
Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:25<00:00,  6.50s/it]
(EngineCore pid=1462) 
(EngineCore pid=1462) INFO 04-07 02:44:26 [default_loader.py:384] Loading weights took 26.10 seconds
(EngineCore pid=1462) INFO 04-07 02:44:26 [gpu_model_runner.py:4820] Model loading took 17.66 GiB memory and 26.884630 seconds
(EngineCore pid=1462) INFO 04-07 02:44:26 [gpu_model_runner.py:5753] Encoder cache will be initialized with a budget of 16384 tokens, and profiled with 1 image items of the maximum feature size.
(EngineCore pid=1462) INFO 04-07 02:44:32 [backends.py:1051] Using cache directory: /root/.cache/vllm/torch_compile_cache/1776bb5122/rank_0_0/backbone for vLLM's torch.compile
(EngineCore pid=1462) INFO 04-07 02:44:32 [backends.py:1111] Dynamo bytecode transform time: 4.61 s
(EngineCore pid=1462) INFO 04-07 02:44:35 [backends.py:372] Cache the graph of compile range (1, 8192) for later use
(EngineCore pid=1462) INFO 04-07 02:44:58 [backends.py:390] Compiling a graph for compile range (1, 8192) takes 25.35 s
(EngineCore pid=1462) INFO 04-07 02:44:59 [decorators.py:640] saved AOT compiled function to /root/.cache/vllm/torch_compile_cache/torch_aot_compile/c303a22a6f3e298f7915fcf64e7ef95c2a38770413b34135cc989ae21cba4a6e/rank_0_0/model
(EngineCore pid=1462) INFO 04-07 02:44:59 [monitor.py:48] torch.compile took 31.11 s in total
(EngineCore pid=1462) /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fla/ops/utils.py:113: UserWarning: Input tensor shape suggests potential format mismatch: seq_len (16) < num_heads (32). This may indicate the inputs were passed in head-first format [B, H, T, ...] when head_first=False was specified. Please verify your input tensor format matches the expected shape [B, T, H, ...].
(EngineCore pid=1462)   return fn(*contiguous_args, **contiguous_kwargs)
(EngineCore pid=1462) /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fla/ops/utils.py:113: UserWarning: Input tensor shape suggests potential format mismatch: seq_len (16) < num_heads (32). This may indicate the inputs were passed in head-first format [B, H, T, ...] when head_first=False was specified. Please verify your input tensor format matches the expected shape [B, T, H, ...].
(EngineCore pid=1462)   return fn(*contiguous_args, **contiguous_kwargs)
(EngineCore pid=1462) INFO 04-07 02:46:12 [monitor.py:76] Initial profiling/warmup run took 73.16 s
(EngineCore pid=1462) INFO 04-07 02:46:18 [kv_cache_utils.py:829] Overriding num_gpu_blocks=0 with num_gpu_blocks_override=512
(EngineCore pid=1462) INFO 04-07 02:46:18 [gpu_model_runner.py:5876] Profiling CUDA graph memory: PIECEWISE=51 (largest=512), FULL=35 (largest=256)
(EngineCore pid=1462) INFO 04-07 02:46:21 [gpu_model_runner.py:5955] Estimated CUDA graph memory: 0.78 GiB total
(EngineCore pid=1462) INFO 04-07 02:46:21 [gpu_worker.py:436] Available KV cache memory: 20.27 GiB
(EngineCore pid=1462) INFO 04-07 02:46:21 [gpu_worker.py:470] In v0.19, CUDA graph memory profiling will be enabled by default (VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1), which more accurately accounts for CUDA graph memory during KV cache allocation. To try it now, set VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 and increase --gpu-memory-utilization from 0.9000 to 0.9176 to maintain the same effective KV cache size.
(EngineCore pid=1462) INFO 04-07 02:46:21 [kv_cache_utils.py:1319] GPU KV cache size: 165,792 tokens
(EngineCore pid=1462) INFO 04-07 02:46:21 [kv_cache_utils.py:1324] Maximum concurrency for 262,144 tokens per request: 2.52x
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%|██████████| 51/51 [00:02<00:00, 21.18it/s]
Capturing CUDA graphs (decode, FULL): 100%|██████████| 35/35 [00:01<00:00, 18.11it/s]
(EngineCore pid=1462) INFO 04-07 02:46:27 [gpu_model_runner.py:6046] Graph capturing finished in 5 secs, took 0.65 GiB
(EngineCore pid=1462) INFO 04-07 02:46:27 [gpu_worker.py:597] CUDA graph pool memory: 0.65 GiB (actual), 0.78 GiB (estimated), difference: 0.13 GiB (20.1%).
(EngineCore pid=1462) INFO 04-07 02:46:27 [core.py:283] init engine (profile, create kv cache, warmup model) took 120.47 seconds
(EngineCore pid=1462) INFO 04-07 02:46:27 [vllm.py:790] Asynchronous scheduling is enabled.
Rendering prompts: 100%
 1/1 [00:00<00:00, 53.01it/s]
Processed prompts: 100%
 1/1 [00:02<00:00,  2.05s/it, est. speed input: 2.44 toks/s, output: 7.80 toks/s]
(EngineCore pid=1462) /usr/local/lib/python3.12/dist-packages/torch/_dynamo/eval_frame.py:1181: UserWarning: Input tensor shape suggests potential format mismatch: seq_len (5) < num_heads (16). This may indicate the inputs were passed in head-first format [B, H, T, ...] Please verify your input tensor format matches the expected shape [B, T, H, ...].
(EngineCore pid=1462)   return fn(*args, **kwargs)
(EngineCore pid=1462) /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fla/ops/utils.py:113: UserWarning: Input tensor shape suggests potential format mismatch: seq_len (5) < num_heads (32). This may indicate the inputs were passed in head-first format [B, H, T, ...] when head_first=False was specified. Please verify your input tensor format matches the expected shape [B, T, H, ...].
(EngineCore pid=1462)   return fn(*contiguous_args, **contiguous_kwargs)
Prompt: 'Hello, my name is', Generated text: ' John. I am a 25-year-old male. I am a student'
```