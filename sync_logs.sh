#!/bin/bash
# 同步远端日志和数据到本地
# 用法: ./sync_logs.sh [--clean]

REMOTE_HOST="zzh@218.192.110.198"
REMOTE_BASE="~/Workspace/260126/agentic-rl"
LOCAL_BASE="."

# 需要同步的目录
SYNC_DIRS=("logger" "local_logger" "stdb")

# 是否清理本地再同步
CLEAN_FIRST=false
if [[ "$1" == "--clean" ]]; then
    CLEAN_FIRST=true
fi

echo "================================================"
echo "   CCAPO 日志同步工具"
echo "================================================"
echo "远程: ${REMOTE_HOST}:${REMOTE_BASE}"
echo "本地: ${LOCAL_BASE}"
echo ""

for DIR in "${SYNC_DIRS[@]}"; do
    echo "-------------------------------------------"
    echo "同步目录: ${DIR}/"
    
    if $CLEAN_FIRST && [[ -d "${LOCAL_BASE}/${DIR}" ]]; then
        echo "  [清理] 删除本地 ${DIR}/"
        rm -rf "${LOCAL_BASE}/${DIR}"
    fi
    
    # 使用 rsync，保留时间戳，增量同步
    # -a: archive mode (递归，保留权限、时间等)
    # -v: verbose
    # -z: 压缩传输
    # --progress: 显示进度
    rsync -avz --progress \
        "${REMOTE_HOST}:${REMOTE_BASE}/${DIR}/" \
        "${LOCAL_BASE}/${DIR}/"
    
    if [[ $? -eq 0 ]]; then
        echo "  [完成] ${DIR}/"
    else
        echo "  [失败] ${DIR}/ - 请检查连接"
    fi
done

echo ""
echo "================================================"
echo "同步完成！"
echo ""
echo "分析日志: python analyze_full_trace.py"
echo "================================================"
