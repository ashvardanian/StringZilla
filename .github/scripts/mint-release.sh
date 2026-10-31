#!/usr/bin/env bash
# Mints a release: stamps the version everywhere, commits, tags, pushes and publishes the release page.
#
#     .github/scripts/mint-release.sh 5.3.3          # stable
#     .github/scripts/mint-release.sh 6.0.0-rc.1     # candidate, published as a prerelease
set -euo pipefail
version=$1 core=${1%%-*}
branch=$(git branch --show-current)
previous=$(git describe --tags --abbrev=0)
git diff --quiet HEAD || { echo "mint-release: commit or stash your changes first"; exit 1; }

IFS=. read -r major minor patch <<< "$core"
export version core major minor patch
echo "$version" > VERSION
perl -pi -e 's/^(version = ")[^"]+/$1$ENV{version}/' Cargo.toml
perl -0pi -e 's/^(name = "stringzilla"\nversion = ")[^"]+/$1$ENV{version}/m' Cargo.lock
perl -pi -e 's/^(  "version": ")[^"]+/$1$ENV{version}/' package.json javascript/@stringzilla-*/package.json
perl -pi -e 's/("\@stringzilla\/[a-z0-9-]+": ")[^"]+/$1$ENV{version}/' package.json
perl -pi -e 's/^version: .+$/version: $ENV{version}/' CITATION.cff
perl -pi -e 's/^(    <Version>)[^<]+/$1$ENV{version}/' StringZilla.csproj
perl -0pi -e 's/^(  <artifactId>stringzilla<\/artifactId>\n  <version>)[^<]+/$1$ENV{version}/m' pom.xml
perl -pi -e 's/^([[:space:]]+VERSION )[0-9.]+$/$1$ENV{core}/' CMakeLists.txt
perl -pi -e 's/^(#define STRINGZILLA_H_VERSION_MAJOR )[0-9]+$/$1$ENV{major}/; s/^(#define STRINGZILLA_H_VERSION_MINOR )[0-9]+$/$1$ENV{minor}/; s/^(#define STRINGZILLA_H_VERSION_PATCH )[0-9]+$/$1$ENV{patch}/' include/stringzilla/stringzilla.h
git commit -am "Release: v$version" # commit-msg rejects it if any site was missed
git tag -a "v$version" -m "Release: v$version"
git push --atomic origin "$branch" "v$version"

# Candidates never take "Latest", and neither does a patch to an older line.
flags=()
[[ $version == *-rc.* ]] && flags+=(--prerelease --title "Candidate v$version")
[ "$(git tag -l 'v*' --sort=-v:refname | grep -v -- - | head -1)" = "v$version" ] || flags+=(--latest=false)
gh release create "v$version" --verify-tag --notes "$(git log --no-merges --format='- %s (%h)' "$previous..v$version^")" "${flags[@]}"
