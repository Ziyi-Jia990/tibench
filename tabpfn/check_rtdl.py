import rtdl
import inspect

print(f"正在检查 rtdl 库，版本为: {rtdl.__version__}")

try:
    # 获取 FTTransformer.__init__ 方法的签名
    sig = inspect.signature(rtdl.FTTransformer.__init__)
    print("\nrtdl.FTTransformer 的构造函数 (__init__) 需要以下参数:")
    print("--------------------------------------------------")
    print(sig)
    print("--------------------------------------------------")

except Exception as e:
    print(f"\n检查时发生错误: {e}")
    print("无法检测 rtdl.FTTransformer。")
