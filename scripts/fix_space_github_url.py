"""
修复 HF Space 中 app.py 的 GitHub URL。
将 Space 中错误的 "your-name" 替换为正确的 "Pyking828"。

使用方法：
    export HF_TOKEN=your_token_here  (Linux/Mac)
    $env:HF_TOKEN="your_token_here"   (Windows PowerShell)
    python scripts/fix_space_github_url.py
"""

import os
from pathlib import Path

from huggingface_hub import HfApi

SPACE_REPO = "Pyking828/eedi-misconception-demo"
LOCAL_APP = Path(__file__).parent.parent / "spaces" / "app.py"


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError(
            "请先设置 HF_TOKEN 环境变量：\n"
            "  PowerShell: $env:HF_TOKEN='hf_xxx...'\n"
            "  Bash:       export HF_TOKEN=hf_xxx..."
        )

    api = HfApi(token=token)

    # 验证 token
    user = api.whoami()
    print(f"已认证为：{user['name']}")

    # 读取本地修复后的 app.py
    content = LOCAL_APP.read_text(encoding="utf-8")
    assert "Pyking828/eedi-misconception-engine" in content, "本地文件中 GitHub URL 不正确！"
    assert "your-name" not in content, "本地文件仍包含 'your-name'！"

    print(f"上传 {LOCAL_APP} → spaces/{SPACE_REPO}/app.py ...")
    api.upload_file(
        path_or_fileobj=content.encode("utf-8"),
        path_in_repo="app.py",
        repo_id=SPACE_REPO,
        repo_type="space",
        commit_message="fix: correct GitHub URL (your-name → Pyking828)",
    )
    print("✓ 上传成功！Space 将在几秒钟内重新构建。")
    print(f"  Space 链接：https://huggingface.co/spaces/{SPACE_REPO}")


if __name__ == "__main__":
    main()
