import torch
import os

accepted_version = {
    '2.2': 'liboptimus_ths-torch2.2-cu121.cpython-310-x86_64-linux-gnu.so',
    '2.3': 'liboptimus_ths-torch2.3-cu121.cpython-310-x86_64-linux-gnu.so',
    '2.5': 'liboptimus_ths-torch2.5-cu124.cpython-310-x86_64-linux-gnu.so',
}

try:
    # args = parse_args()
    version = '.'.join(torch.__version__.split('.')[:2])
    if version in accepted_version:
        # 修改后的代码行
        torch.ops.load_library(
            os.path.join(
                os.getenv("STEPVIDEO_LIB_ROOT", "/file_system/models/dit/stepvideo-t2v/"),  # 优先读取环境变量
                "lib",  # 固定子目录
                accepted_version[version]  # 版本对应的文件名
            )
        )
    else:
        raise ValueError("Not supported torch version for liboptimus")
except Exception as err:
    print(err)
