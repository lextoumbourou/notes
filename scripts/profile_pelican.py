"""Measure Pelican stages and individual source files without changing settings.

Run from the site root, using the site's Python environment:
    ENV=local PYTHONPATH=. .venv/bin/python scripts/profile_pelican.py \
        --report /tmp/pelican.json --output /tmp/pelican-output
Add --cprofile /tmp/pelican.prof for detailed function profiling. Profiled
timings include instrumentation overhead and are not benchmark results.
Extra Pelican arguments go after -- (for example --ignore-cache).
"""

import argparse
import cProfile
import functools
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cprofile", type=Path)
    parser.add_argument("pelican_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    extra = args.pelican_args
    if extra[:1] == ["--"]:
        extra = extra[1:]

    stages = []
    files = []
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "profiled": bool(args.cprofile),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("pelican", "Markdown", "Pygments", "beautifulsoup4", "PyYAML")
        },
        "stages": stages,
        "files": files,
    }
    profiler = cProfile.Profile() if args.cprofile else None
    started = time.perf_counter()
    if profiler:
        profiler.enable()
    try:
        import pelican
        from pelican import generators, readers

        def wrap_stage(cls, name):
            original = getattr(cls, name)

            @functools.wraps(original)
            def timed(*positional, **kwargs):
                begin = time.perf_counter()
                try:
                    return original(*positional, **kwargs)
                finally:
                    stages.append({
                        "stage": f"{cls.__name__}.{name}",
                        "seconds": time.perf_counter() - begin,
                    })

            setattr(cls, name, timed)

        for cls in (generators.ArticlesGenerator, generators.PagesGenerator,
                    generators.StaticGenerator):
            for name in ("generate_context", "generate_output"):
                wrap_stage(cls, name)

        original_read = readers.Readers.read_file

        @functools.wraps(original_read)
        def timed_read(self, base_path, path, *positional, **kwargs):
            begin = time.perf_counter()
            try:
                return original_read(self, base_path, path, *positional, **kwargs)
            finally:
                files.append({"path": str(path), "seconds": time.perf_counter() - begin})

        readers.Readers.read_file = timed_read
        command = ["./notes/", f"--output={args.output}", *extra]
        report["arguments"] = command
        pelican.main(command)
        report["completed"] = True
    finally:
        report["seconds"] = time.perf_counter() - started
        files.sort(key=lambda item: item["seconds"], reverse=True)
        if profiler:
            profiler.disable()
            args.cprofile.parent.mkdir(parents=True, exist_ok=True)
            profiler.dump_stats(str(args.cprofile))
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Saved timings to {args.report}")


if __name__ == "__main__":
    main()
