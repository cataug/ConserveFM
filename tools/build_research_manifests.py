#!/usr/bin/env python3

from __future__ import annotations

import csv
import gzip
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr


ROOT = Path.home() / "ConserveFM"

ERA5 = ROOT / "data/ERA5_WeatherBench2_1979_2022"
GHCN = ROOT / "data/GHCN_Daily_by_year"
ISD = ROOT / "data/ISD_2023_stations"

META = ROOT / "data/station_metadata"

GHCN_STATIONS = META / "ghcnd-stations.txt"
GHCN_INVENTORY = META / "ghcnd-inventory.txt"
ISD_HISTORY = META / "isd-history.csv"

OUT = ROOT / "manifests/research_v1"

SPLITS = {
    "train": (
        np.datetime64("1979-01-01T00:00:00"),
        np.datetime64("2019-01-01T00:00:00"),
    ),
    "val": (
        np.datetime64("2019-01-01T00:00:00"),
        np.datetime64("2021-01-01T00:00:00"),
    ),
    "test": (
        np.datetime64("2021-01-01T00:00:00"),
        np.datetime64("2023-01-01T00:00:00"),
    ),
}

CONTEXT_STEPS = int(
    os.environ.get(
        "CONSERVEFM_CONTEXT_STEPS",
        "4",
    )
)

LEAD_HOURS = [
    int(x)
    for x in os.environ.get(
        "CONSERVEFM_LEAD_HOURS",
        "6,24,72",
    ).split(",")
]

BUILD_GHCN_OBS = (
    os.environ.get(
        "BUILD_GHCN_OBS",
        "0",
    ) == "1"
)

# Station variables useful for independent validation.
GHCN_ELEMENTS = {
    "PRCP": {
        "units": "mm",
        "scale": 0.1,
        "era5_target":
            "daily_sum(total_precipitation_6hr)",
    },
    "TMAX": {
        "units": "degC",
        "scale": 0.1,
        "era5_target":
            "daily_max(2m_temperature)",
    },
    "TMIN": {
        "units": "degC",
        "scale": 0.1,
        "era5_target":
            "daily_min(2m_temperature)",
    },
    "TAVG": {
        "units": "degC",
        "scale": 0.1,
        "era5_target":
            "daily_mean(2m_temperature)",
    },
    "AWND": {
        "units": "m/s",
        "scale": 0.1,
        "era5_target":
            "daily_mean(10m_wind_speed)",
    },
    "ASLP": {
        "units": "hPa",
        "scale": 0.1,
        "era5_target":
            "daily_mean(mean_sea_level_pressure)",
    },
    "ASTP": {
        "units": "hPa",
        "scale": 0.1,
        "era5_target":
            "daily_mean(surface_pressure)",
    },
}


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = list(rows)

    if not rows:
        path.write_text(
            "",
            encoding="utf-8",
        )
        return

    if fieldnames is None:
        fieldnames = list(rows[0].keys())

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def split_for_time(t):
    for split, (start, end) in SPLITS.items():
        if start <= t < end:
            return split

    return None


def circular_lon_distance(grid_lon, lon):
    return np.abs(
        (
            (grid_lon - lon + 180.0)
            % 360.0
        )
        - 180.0
    )


def haversine_km(
    lat1,
    lon1,
    lat2,
    lon2,
):
    r = 6371.0088

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)

    dp = math.radians(
        lat2 - lat1
    )

    dl = math.radians(
        (
            (lon2 - lon1 + 180.0)
            % 360.0
        )
        - 180.0
    )

    a = (
        math.sin(dp / 2) ** 2
        +
        math.cos(p1)
        * math.cos(p2)
        * math.sin(dl / 2) ** 2
    )

    return (
        2
        * r
        * math.asin(
            min(1.0, math.sqrt(a))
        )
    )


def nearest_grid(
    station_lat,
    station_lon,
    latitudes,
    longitudes,
):
    lat_idx = int(
        np.argmin(
            np.abs(
                latitudes - station_lat
            )
        )
    )

    lon_idx = int(
        np.argmin(
            circular_lon_distance(
                longitudes,
                station_lon,
            )
        )
    )

    grid_lat = float(
        latitudes[lat_idx]
    )

    grid_lon = float(
        longitudes[lon_idx]
    )

    distance = haversine_km(
        station_lat,
        station_lon,
        grid_lat,
        grid_lon,
    )

    return (
        lat_idx,
        lon_idx,
        grid_lat,
        grid_lon,
        distance,
    )


def open_era5():
    print(
        "[ERA5] opening local Zarr metadata"
    )

    ds = xr.open_zarr(
        str(ERA5),
        consolidated=False,
        chunks=None,
    )

    times = ds.time.values
    latitudes = ds.latitude.values
    longitudes = ds.longitude.values

    return (
        ds,
        times,
        latitudes,
        longitudes,
    )


def build_era5_time_manifest(times):
    print(
        "[ERA5] building time/split manifest"
    )

    rows = []

    split_counts = defaultdict(int)

    for idx, t in enumerate(times):
        split = split_for_time(t)

        if split is None:
            continue

        ts = np.datetime_as_string(
            t,
            unit="s",
        )

        year = int(ts[:4])
        month = int(ts[5:7])

        rows.append({
            "time_index": idx,
            "timestamp": ts,
            "year": year,
            "month": month,
            "split": split,
        })

        split_counts[split] += 1

    write_csv(
        OUT / "era5_time_split.csv",
        rows,
    )

    return split_counts


def build_era5_examples(times):
    print(
        "[ERA5] building forecasting examples"
    )

    selected = [
        i
        for i, t in enumerate(times)
        if split_for_time(t) is not None
    ]

    if len(selected) < 2:
        raise RuntimeError(
            "Not enough ERA5 timestamps"
        )

    diffs = np.diff(
        times[selected]
    ).astype(
        "timedelta64[s]"
    ).astype(int)

    step_seconds = int(
        np.median(diffs)
    )

    step_hours = (
        step_seconds / 3600.0
    )

    print(
        f"[ERA5] timestep = "
        f"{step_hours:g} h"
    )

    lead_steps = {}

    for hours in LEAD_HOURS:
        steps = hours / step_hours

        if not float(steps).is_integer():
            raise RuntimeError(
                f"Lead {hours} h is not "
                f"compatible with "
                f"{step_hours:g} h timestep"
            )

        lead_steps[hours] = int(
            steps
        )

    rows = []
    counts = defaultdict(int)

    for split, (start, end) in SPLITS.items():
        indices = np.where(
            (times >= start)
            & (times < end)
        )[0]

        if len(indices) == 0:
            continue

        for pos in range(
            CONTEXT_STEPS - 1,
            len(indices),
        ):
            context_end_idx = int(
                indices[pos]
            )

            context_start_idx = int(
                indices[
                    pos
                    - CONTEXT_STEPS
                    + 1
                ]
            )

            for hours, steps in (
                lead_steps.items()
            ):
                target_pos = (
                    pos + steps
                )

                if target_pos >= len(indices):
                    continue

                target_idx = int(
                    indices[target_pos]
                )

                sample_id = (
                    f"{split}_"
                    f"{context_end_idx:06d}_"
                    f"lead{hours:03d}"
                )

                rows.append({
                    "sample_id":
                        sample_id,
                    "split":
                        split,
                    "context_steps":
                        CONTEXT_STEPS,
                    "context_start_idx":
                        context_start_idx,
                    "context_end_idx":
                        context_end_idx,
                    "target_idx":
                        target_idx,
                    "context_start":
                        np.datetime_as_string(
                            times[
                                context_start_idx
                            ],
                            unit="s",
                        ),
                    "context_end":
                        np.datetime_as_string(
                            times[
                                context_end_idx
                            ],
                            unit="s",
                        ),
                    "target_time":
                        np.datetime_as_string(
                            times[target_idx],
                            unit="s",
                        ),
                    "lead_hours":
                        hours,
                })

                counts[split] += 1

    write_csv(
        OUT / "era5_examples.csv",
        rows,
    )

    return counts


def read_ghcn_station_metadata():
    if not GHCN_STATIONS.exists():
        raise FileNotFoundError(
            GHCN_STATIONS
        )

    result = {}

    with GHCN_STATIONS.open(
        encoding="ascii",
        errors="replace",
    ) as f:
        for line in f:
            station_id = (
                line[0:11].strip()
            )

            if not station_id:
                continue

            try:
                lat = float(
                    line[12:20]
                )
                lon = float(
                    line[21:30]
                )
                elevation = float(
                    line[31:37]
                )
            except ValueError:
                continue

            result[station_id] = {
                "station_id":
                    station_id,
                "latitude":
                    lat,
                "longitude":
                    lon,
                "elevation_m":
                    elevation,
                "state":
                    line[38:40].strip(),
                "name":
                    line[41:71].strip(),
                "gsn_flag":
                    line[72:75].strip(),
                "hcn_crn_flag":
                    line[76:79].strip(),
                "wmo_id":
                    line[80:85].strip(),
            }

    return result


def read_ghcn_inventory():
    useful_ids = set()
    inventory_rows = []

    with GHCN_INVENTORY.open(
        encoding="ascii",
        errors="replace",
    ) as f:
        for line in f:
            station_id = (
                line[0:11].strip()
            )

            element = (
                line[31:35].strip()
            )

            try:
                first_year = int(
                    line[36:40]
                )
                last_year = int(
                    line[41:45]
                )
            except ValueError:
                continue

            if (
                element
                not in GHCN_ELEMENTS
            ):
                continue

            if (
                last_year < 1979
                or first_year > 2023
            ):
                continue

            useful_ids.add(
                station_id
            )

            inventory_rows.append({
                "station_id":
                    station_id,
                "element":
                    element,
                "first_year":
                    first_year,
                "last_year":
                    last_year,
            })

    return (
        useful_ids,
        inventory_rows,
    )


def build_ghcn_grid(
    latitudes,
    longitudes,
):
    print(
        "[GHCN] mapping stations "
        "to ERA5 grid"
    )

    metadata = (
        read_ghcn_station_metadata()
    )

    useful_ids, inventory = (
        read_ghcn_inventory()
    )

    rows = []
    lookup = {}

    for station_id in sorted(
        useful_ids
    ):
        station = metadata.get(
            station_id
        )

        if station is None:
            continue

        (
            lat_idx,
            lon_idx,
            grid_lat,
            grid_lon,
            distance,
        ) = nearest_grid(
            station["latitude"],
            station["longitude"],
            latitudes,
            longitudes,
        )

        row = {
            **station,
            "era5_lat_idx":
                lat_idx,
            "era5_lon_idx":
                lon_idx,
            "era5_grid_lat":
                grid_lat,
            "era5_grid_lon":
                grid_lon,
            "distance_to_grid_km":
                round(distance, 3),
            "era5_grid_id":
                f"{lat_idx}:{lon_idx}",
        }

        rows.append(row)
        lookup[station_id] = row

    write_csv(
        OUT / "ghcn_station_era5_grid.csv",
        rows,
    )

    write_csv(
        OUT / "ghcn_inventory.csv",
        inventory,
    )

    print(
        f"[GHCN] mapped stations: "
        f"{len(rows):,}"
    )

    return lookup


def build_ghcn_year_splits():
    rows = []

    for year in range(
        1979,
        2024,
    ):
        if year <= 2018:
            split = "train"
            overlap = True

        elif year <= 2020:
            split = "val"
            overlap = True

        elif year <= 2022:
            split = "test"
            overlap = True

        else:
            split = (
                "external_2023_no_era5"
            )
            overlap = False

        path = (
            GHCN
            / f"{year}.csv.gz"
        )

        rows.append({
            "year": year,
            "split": split,
            "era5_temporal_overlap":
                overlap,
            "path": str(path),
            "present":
                path.exists(),
            "size_bytes":
                (
                    path.stat().st_size
                    if path.exists()
                    else 0
                ),
        })

    write_csv(
        OUT / "ghcn_year_splits.csv",
        rows,
    )


def build_ghcn_observations(
    station_lookup,
):
    if not BUILD_GHCN_OBS:
        print(
            "[GHCN] observation-level "
            "manifest skipped"
        )
        return 0

    out_path = (
        OUT
        / "ghcn_val_test_observations.csv.gz"
    )

    fields = [
        "split",
        "date",
        "station_id",
        "station_name",
        "station_lat",
        "station_lon",
        "era5_lat_idx",
        "era5_lon_idx",
        "era5_grid_lat",
        "era5_grid_lon",
        "distance_to_grid_km",
        "element",
        "value",
        "units",
        "era5_target",
        "m_flag",
        "q_flag",
        "s_flag",
        "obs_time",
    ]

    count = 0

    with gzip.open(
        out_path,
        "wt",
        newline="",
        encoding="utf-8",
    ) as out:
        writer = csv.DictWriter(
            out,
            fieldnames=fields,
        )
        writer.writeheader()

        for year in range(
            2019,
            2023,
        ):
            split = (
                "val"
                if year <= 2020
                else "test"
            )

            path = (
                GHCN
                / f"{year}.csv.gz"
            )

            print(
                f"[GHCN] parsing {year} "
                f"for {split}"
            )

            with gzip.open(
                path,
                "rt",
                encoding="ascii",
                errors="replace",
                newline="",
            ) as src:
                reader = csv.reader(src)

                for raw in reader:
                    if len(raw) < 8:
                        continue

                    (
                        station_id,
                        date,
                        element,
                        raw_value,
                        m_flag,
                        q_flag,
                        s_flag,
                        obs_time,
                    ) = raw[:8]

                    if (
                        element
                        not in GHCN_ELEMENTS
                    ):
                        continue

                    # Only observations without
                    # a GHCN quality flag.
                    if q_flag.strip():
                        continue

                    station = (
                        station_lookup.get(
                            station_id
                        )
                    )

                    if station is None:
                        continue

                    try:
                        raw_value = int(
                            raw_value
                        )
                    except ValueError:
                        continue

                    cfg = (
                        GHCN_ELEMENTS[
                            element
                        ]
                    )

                    value = (
                        raw_value
                        * cfg["scale"]
                    )

                    writer.writerow({
                        "split":
                            split,
                        "date":
                            date,
                        "station_id":
                            station_id,
                        "station_name":
                            station["name"],
                        "station_lat":
                            station[
                                "latitude"
                            ],
                        "station_lon":
                            station[
                                "longitude"
                            ],
                        "era5_lat_idx":
                            station[
                                "era5_lat_idx"
                            ],
                        "era5_lon_idx":
                            station[
                                "era5_lon_idx"
                            ],
                        "era5_grid_lat":
                            station[
                                "era5_grid_lat"
                            ],
                        "era5_grid_lon":
                            station[
                                "era5_grid_lon"
                            ],
                        "distance_to_grid_km":
                            station[
                                "distance_to_grid_km"
                            ],
                        "element":
                            element,
                        "value":
                            value,
                        "units":
                            cfg["units"],
                        "era5_target":
                            cfg[
                                "era5_target"
                            ],
                        "m_flag":
                            m_flag,
                        "q_flag":
                            q_flag,
                        "s_flag":
                            s_flag,
                        "obs_time":
                            obs_time,
                    })

                    count += 1

                    if count % 1_000_000 == 0:
                        print(
                            f"[GHCN] "
                            f"{count:,} "
                            "observations"
                        )

    print(
        f"[GHCN] val/test "
        f"observations: {count:,}"
    )

    return count


def normalize_isd_id(
    usaf,
    wban,
):
    usaf = (
        str(usaf)
        .strip()
        .replace('"', "")
    )

    wban = (
        str(wban)
        .strip()
        .replace('"', "")
    )

    if usaf.endswith(".0"):
        usaf = usaf[:-2]

    if wban.endswith(".0"):
        wban = wban[:-2]

    return (
        usaf.zfill(6)
        + wban.zfill(5)
    )


def build_isd_grid(
    latitudes,
    longitudes,
):
    print(
        "[ISD] mapping downloaded "
        "2023 stations to ERA5 grid"
    )

    downloaded = {
        p.stem: p
        for p in ISD.glob(
            "*.csv"
        )
    }

    rows = []
    matched = set()

    with ISD_HISTORY.open(
        newline="",
        encoding="utf-8-sig",
        errors="replace",
    ) as f:
        reader = csv.DictReader(f)

        for original in reader:
            row = {
                (k or "").strip():
                    (v or "").strip()
                for k, v
                in original.items()
            }

            station_id = normalize_isd_id(
                row.get("USAF", ""),
                row.get("WBAN", ""),
            )

            if (
                station_id
                not in downloaded
            ):
                continue

            try:
                lat = float(
                    row.get(
                        "LAT",
                        row.get(
                            "LATITUDE",
                            "",
                        ),
                    )
                )

                lon = float(
                    row.get(
                        "LON",
                        row.get(
                            "LONGITUDE",
                            "",
                        ),
                    )
                )

            except ValueError:
                continue

            (
                lat_idx,
                lon_idx,
                grid_lat,
                grid_lon,
                distance,
            ) = nearest_grid(
                lat,
                lon,
                latitudes,
                longitudes,
            )

            path = downloaded[
                station_id
            ]

            rows.append({
                "station_id":
                    station_id,
                "usaf":
                    row.get(
                        "USAF",
                        "",
                    ),
                "wban":
                    row.get(
                        "WBAN",
                        "",
                    ),
                "name":
                    row.get(
                        "STATION NAME",
                        row.get(
                            "STATION_NAME",
                            "",
                        ),
                    ),
                "country":
                    row.get(
                        "CTRY",
                        "",
                    ),
                "state":
                    row.get(
                        "STATE",
                        "",
                    ),
                "icao":
                    row.get(
                        "ICAO",
                        "",
                    ),
                "latitude":
                    lat,
                "longitude":
                    lon,
                "elevation_m":
                    row.get(
                        "ELEV(M)",
                        "",
                    ),
                "era5_lat_idx":
                    lat_idx,
                "era5_lon_idx":
                    lon_idx,
                "era5_grid_lat":
                    grid_lat,
                "era5_grid_lon":
                    grid_lon,
                "distance_to_grid_km":
                    round(
                        distance,
                        3,
                    ),
                "era5_grid_id":
                    f"{lat_idx}:{lon_idx}",
                "split":
                    "external_2023_no_era5",
                "era5_temporal_overlap":
                    False,
                "file":
                    str(path),
                "size_bytes":
                    path.stat().st_size,
            })

            matched.add(
                station_id
            )

    write_csv(
        OUT
        / "isd2023_station_era5_grid.csv",
        rows,
    )

    unmatched = []

    for station_id, path in (
        sorted(
            downloaded.items()
        )
    ):
        if station_id in matched:
            continue

        unmatched.append({
            "station_id":
                station_id,
            "file":
                str(path),
            "size_bytes":
                path.stat().st_size,
        })

    write_csv(
        OUT
        / "isd2023_unmatched_stations.csv",
        unmatched,
    )

    print(
        f"[ISD] downloaded: "
        f"{len(downloaded):,}"
    )

    print(
        f"[ISD] metadata matched: "
        f"{len(rows):,}"
    )

    print(
        f"[ISD] unmatched: "
        f"{len(unmatched):,}"
    )

    return (
        len(rows),
        len(unmatched),
    )


def write_protocol():
    protocol = {
        "name":
            "ConserveFM research manifest v1",

        "generated_at":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "era5": {
            "period":
                "1979-01-01/2022-12-31",
            "resolution":
                "6-hourly",
            "grid":
                "240x121, 1.5 degree",
            "variables":
                11,
            "pressure_levels":
                13,
        },

        "splits": {
            "train":
                "1979-01-01/2018-12-31",
            "val":
                "2019-01-01/2020-12-31",
            "test":
                "2021-01-01/2022-12-31",
        },

        "forecast_examples": {
            "context_steps":
                CONTEXT_STEPS,
            "lead_hours":
                LEAD_HOURS,
        },

        "ghcn": {
            "downloaded":
                "1979-2023",
            "paired_with_era5":
                "1979-2022",
            "observation_eval":
                "2019-2022",
            "elements":
                list(
                    GHCN_ELEMENTS.keys()
                ),
        },

        "isd": {
            "downloaded":
                "2023",
            "era5_overlap":
                False,
            "role":
                (
                    "external observation set; "
                    "spatially mapped now, "
                    "temporal pairing requires "
                    "ERA5 2023"
                ),
        },
    }

    (
        OUT / "protocol.json"
    ).write_text(
        json.dumps(
            protocol,
            indent=2,
        ),
        encoding="utf-8",
    )


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 74)
    print(
        "ConserveFM research-ready "
        "manifest builder"
    )
    print("=" * 74)

    (
        ds,
        times,
        latitudes,
        longitudes,
    ) = open_era5()

    split_counts = (
        build_era5_time_manifest(
            times
        )
    )

    example_counts = (
        build_era5_examples(
            times
        )
    )

    ghcn_lookup = (
        build_ghcn_grid(
            latitudes,
            longitudes,
        )
    )

    build_ghcn_year_splits()

    ghcn_obs = (
        build_ghcn_observations(
            ghcn_lookup
        )
    )

    (
        isd_matched,
        isd_unmatched,
    ) = build_isd_grid(
        latitudes,
        longitudes,
    )

    write_protocol()

    summary = {
        "generated_at":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "era5_timestamps":
            dict(split_counts),

        "era5_examples":
            dict(example_counts),

        "ghcn_mapped_stations":
            len(ghcn_lookup),

        "ghcn_val_test_observations":
            ghcn_obs,

        "isd2023_matched_stations":
            isd_matched,

        "isd2023_unmatched_stations":
            isd_unmatched,
    }

    (
        OUT / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 74)
    print("DONE")
    print("=" * 74)

    for path in sorted(
        OUT.iterdir()
    ):
        print(
            f"{path.name:<40} "
            f"{path.stat().st_size / 1024 / 1024:8.2f} MiB"
        )

    print()
    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
