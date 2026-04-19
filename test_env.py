# test_cuda_complete.py
import torch
import triton

print("=" * 50)
print("CUDA 环境验证")
print("=" * 50)

# 1. 检查 CUDA 是否可用
print(f"✓ CUDA available: {torch.cuda.is_available()}")

# 2. 检查 GPU 信息
if torch.cuda.is_available():
    print(f"✓ GPU Count: {torch.cuda.device_count()}")
    print(f"✓ GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"✓ CUDA Version (PyTorch): {torch.version.cuda}")
    print(f"✓ cuDNN Version: {torch.backends.cudnn.version()}")

# 3. 检查 Triton
print(f"✓ Triton Version: {triton.__version__}")

# 4. 简单计算测试
x = torch.randn(100, 100, device='cuda')
y = torch.randn(100, 100, device='cuda')
z = x @ y  # 矩阵乘法
print(f"✓ CUDA 计算测试：{z.shape}")

print("=" * 50)
print("所有检查通过！环境完整！🎉")