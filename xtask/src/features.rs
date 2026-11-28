//! The `diagnostics` feature graph, statically: a diagnostics build of the
//! binary has every diagnostic path compiled in, a diagnostics build of any
//! other crate has every path of the crates it is built from, and a plain
//! build has none at all.
//!
//! The failure this exists to prevent is a silent one. A crate gates code on
//! its `diagnostics` feature, the build's feature never reaches it, and the
//! switch reads as inert on hardware with nothing to say so. See
//! docs/diagnostics-builds.md.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use toml::Value;
use walkdir::WalkDir;

/// The package a diagnostics build is made for.
const BINARY: &str = "victory";

/// The feature, in every crate that has one.
const FEATURE: &str = "diagnostics";

/// The source of every crate that gates anything on the feature.
const GATE: &str = "feature = \"diagnostics\"";

/// Crates that are no binary's diagnostic path: the checks themselves mention
/// the feature in prose, and an unforwarded name here would read as a gate.
const MAINTENANCE: &[&str] = &["xtask"];

/// Comparisons of `{crate}: {dependency}` whose forwarding is deliberately
/// left out, with the reason. Keep this as short as the reasons are good.
const SPARED: &[(&str, &str)] = &[(
    "victory-gpu: victory-shader",
    "shader diagnostics changes what the emitter produces (the loop limit) and the translation \
     fingerprint the shader cache keys on, so forwarding it would change what a victory-gpu \
     diagnostics build measures",
)];

struct Package {
    name: String,
    manifest: PathBuf,
    /// Feature name to what it enables, as written (`victory-gpu/diagnostics`).
    features: BTreeMap<String, Vec<String>>,
    /// Crates this one depends on, by package name.
    depends: BTreeSet<String>,
    /// `features = [...]` asked for on a dependency: enabled in every build.
    unconditional: Vec<String>,
    /// Whether this crate's source gates anything on the feature.
    gated: bool,
}

/// Checks the graph and prints it. Fails with what to fix when it is wrong.
pub fn run(root: &Path) -> Result<()> {
    let packages = packages(root)?;
    let reach = reachable(&packages, BINARY);

    println!(
        "{:<18} {:<7} {:<8} diagnostics enables",
        "crate", "gates", "reached"
    );
    for p in &packages {
        println!(
            "{:<18} {:<7} {:<8} {}",
            p.name,
            yes_no(p.gated),
            yes_no(reach.contains(&p.name)),
            p.features
                .get(FEATURE)
                .map(|f| f.join(", "))
                .unwrap_or_default()
        );
    }

    // A crate that compiles diagnostic paths in must be in every diagnostics
    // build that includes it, or those paths are dead in the build that runs.
    let mut missing = Vec::new();
    for p in &packages {
        // A crate with no `diagnostics` feature of its own has nothing to
        // forward into; the binary-level check above still requires its paths
        // in the binary's build.
        if !p.features.contains_key(FEATURE) || MAINTENANCE.contains(&p.name.as_str()) {
            continue;
        }
        for dep in &p.depends {
            let gated = dep != &p.name && packages.iter().any(|d| &d.name == dep && d.gated);
            if gated && !forwards(p, dep) {
                if SPARED.iter().any(|(pair, _)| {
                    let mut parts = pair.split(": ");
                    parts.next() == Some(p.name.as_str()) && parts.next() == Some(dep.as_str())
                }) {
                    continue;
                }
                missing.push(format!(
                    "`{}` ({}): add `\"{}/{FEATURE}\"` to its `{FEATURE}` feature",
                    p.name,
                    p.manifest.display(),
                    dep
                ));
            }
        }
    }
    if !missing.is_empty() {
        missing.sort();
        bail!(
            "these crates gate on `cfg(feature = \"{FEATURE}\")` and no build of their dependent \
             compiles them in:\n  {}",
            missing.join("\n  ")
        );
    }

    // The binary's own build: every crate that gates on the feature at all.
    let unreached: Vec<&Package> = packages
        .iter()
        .filter(|p| {
            p.name != BINARY
                && p.gated
                && !MAINTENANCE.contains(&p.name.as_str())
                && !reach.contains(&p.name)
        })
        .collect();
    if !unreached.is_empty() {
        bail!(
            "a diagnostics build of `{BINARY}` does not compile every crate that gates on it in: \
             {}",
            unreached
                .iter()
                .map(|p| format!("`{}` needs a forwarding path from `{BINARY}`", p.name))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    // The other half of the same invariant: a plain build must not have a
    // single diagnostic path in it, whatever profile it is built with.
    let mut leaks = Vec::new();
    for p in &packages {
        if p.features
            .get("default")
            .is_some_and(|d| d.iter().any(|f| f == FEATURE))
        {
            leaks.push(format!("`{}` enables it in `default`", p.name));
        }
        for f in &p.unconditional {
            leaks.push(format!("`{}` asks for `{f}` unconditionally", p.name));
        }
    }
    if !leaks.is_empty() {
        bail!(
            "`{FEATURE}` must be off unless asked for by name, but:\n  {}",
            leaks.join("\n  ")
        );
    }

    println!(
        "\n{FEATURE}: {} crate(s) gate on it, every one of them reached from `{BINARY}`, and no \
         plain build enables it",
        packages.iter().filter(|p| p.gated).count()
    );
    Ok(())
}

fn yes_no(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
}

/// Whether `p`'s `diagnostics` feature names `dep/diagnostics`.
fn forwards(p: &Package, dep: &str) -> bool {
    let wanted = format!("{dep}/{FEATURE}");
    p.features
        .get(FEATURE)
        .is_some_and(|enables| enables.contains(&wanted))
}

/// Every crate `from`'s `diagnostics` feature turns on, by forwarding.
fn reachable(packages: &[Package], from: &str) -> BTreeSet<String> {
    let by_name: BTreeMap<&str, &Package> = packages.iter().map(|p| (p.name.as_str(), p)).collect();
    let dep_feature = format!("/{FEATURE}");
    let mut seen = BTreeSet::new();
    let mut stack = vec![from.to_string()];
    while let Some(name) = stack.pop() {
        if !seen.insert(name.clone()) {
            continue;
        }
        let Some(package) = by_name.get(name.as_str()) else {
            continue;
        };
        // `victory-gpu/diagnostics` forwards; `dep:windows-core` and
        // `windows/Win32_Media_Audio` name no crate of ours.
        stack.extend(
            package
                .features
                .get(FEATURE)
                .into_iter()
                .flatten()
                .filter_map(|f| f.strip_suffix(dep_feature.as_str()))
                .map(str::to_string),
        );
    }
    seen
}

fn packages(root: &Path) -> Result<Vec<Package>> {
    let mut out = Vec::new();
    for manifest in manifests(root) {
        let text = std::fs::read_to_string(&manifest)
            .with_context(|| format!("reading {}", manifest.display()))?;
        let value: Value =
            toml::from_str(&text).with_context(|| format!("parsing {}", manifest.display()))?;
        let Some(name) = value
            .get("package")
            .and_then(|p| p.get("name"))
            .and_then(Value::as_str)
        else {
            // The workspace manifest has no [package].
            continue;
        };
        let features = value
            .get("features")
            .and_then(Value::as_table)
            .map(|t| {
                t.iter()
                    .map(|(name, listed)| (name.clone(), strings(listed)))
                    .collect()
            })
            .unwrap_or_default();
        let (depends, unconditional) = dependencies(&value);
        let dir = manifest.parent().unwrap_or(root).to_path_buf();
        let gated = gates(&dir)?;
        out.push(Package {
            name: name.to_string(),
            manifest,
            features,
            depends,
            unconditional,
            gated,
        });
    }
    Ok(out)
}

fn manifests(root: &Path) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = WalkDir::new(root)
        .max_depth(3)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_name() == "Cargo.toml")
        .map(walkdir::DirEntry::into_path)
        .filter(|path| {
            // `target/` holds dependency manifests, and `tests/fixtures` are
            // standalone crates built on demand rather than members.
            path.strip_prefix(root)
                .is_ok_and(|rel| !rel.starts_with("target") && !rel.starts_with("tests"))
        })
        .collect();
    found.sort();
    found
}

fn strings(value: &Value) -> Vec<String> {
    value
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// The crates a manifest depends on, and the features it asks of them by name.
fn dependencies(value: &Value) -> (BTreeSet<String>, Vec<String>) {
    let mut depends = BTreeSet::new();
    let mut unconditional = Vec::new();
    let mut tables: Vec<&Value> = Vec::new();
    for key in ["dependencies", "dev-dependencies", "build-dependencies"] {
        tables.extend(value.get(key));
    }
    for target in value
        .get("target")
        .and_then(Value::as_table)
        .into_iter()
        .flatten()
    {
        let Some(under) = target.1.as_table() else {
            continue;
        };
        for key in ["dependencies", "dev-dependencies", "build-dependencies"] {
            tables.extend(under.get(key));
        }
    }
    for (key, entry) in tables.iter().filter_map(|t| t.as_table()).flatten() {
        let name = entry
            .get("package")
            .and_then(Value::as_str)
            .unwrap_or(key)
            .to_string();
        depends.insert(name.clone());
        let Some(asked) = entry.get("features").and_then(Value::as_array) else {
            continue;
        };
        for feature in asked.iter().filter_map(Value::as_str) {
            if feature == FEATURE || feature.ends_with(&format!("/{FEATURE}")) {
                unconditional.push(format!("{name}/{feature}"));
            }
        }
    }
    (depends, unconditional)
}

/// Whether any source file under `dir` gates on the feature.
fn gates(dir: &Path) -> Result<bool> {
    let sources = WalkDir::new(dir.join("src"))
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().is_some_and(|x| x == "rs"));
    for entry in sources {
        let text = std::fs::read_to_string(entry.path())
            .with_context(|| format!("reading {}", entry.path().display()))?;
        if text.contains(GATE) {
            return Ok(true);
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    #[test]
    fn repository_forwards_diagnostics() {
        super::run(&crate::workspace_root()).unwrap();
    }
}
