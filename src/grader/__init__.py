"""dao-fa-grader：道法主观题智能判卷服务。"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("dao-fa-grader")
except PackageNotFoundError:  # 未以 pip 安装直接运行源码时
    __version__ = "0.0.0+uninstalled"
