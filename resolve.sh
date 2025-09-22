# 持续解决 README.md 冲突为本地版本，并继续 rebase，直到成功结束
while ! git rebase --continue 2>/dev/null; do
    git checkout --theirs README.md
    git add README.md
done
