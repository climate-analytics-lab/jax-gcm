"""Site profiles — everything about a cluster that is not generic Kubernetes.

(Named ``sites.py``, not ``site.py``: Python imports a stdlib module called
``site`` at interpreter startup, so a local ``site.py`` is already shadowed in
``sys.modules`` by the time any path manipulation runs.)

The Job *shape* (clone pinned SHAs, run jcm, write to a volume, survive or
refuse to survive failure) is portable. What is not portable is roughly
fifteen values, and they are collected here so a second cluster is a new
dict rather than a fork of the generators.

The boundary is sharper than it looks. On Nautilus the nodes advertise
THREE GPU resource names:

    nvidia.com/gpu       the STANDARD Kubernetes name — GKE/EKS use this
    nvidia.com/a100      NRP's quota bucket — the one we must request here
    nvidia.com/a100-80g  an NRP variant

Only ``nvidia.com/a100`` is quota-tracked in our namespace, so requesting
the standard name would bypass the quota and land anywhere. A cloud profile
would use ``nvidia.com/gpu`` plus a provider-specific accelerator label.

No cloud profile is included. Writing one without a cluster to test it
against would be guesswork, and untested platform code that *looks*
authoritative is worse than none — the template below records what a new
profile has to supply.
"""

from __future__ import annotations

NAUTILUS = {
    "name": "nautilus",
    "namespace": "climate-analytics",
    # Extended resource to request. NOT nvidia.com/gpu here — see above.
    "gpu_resource": "nvidia.com/a100",
    # Selects any 80GB A100. The quota bucket also covers a 40GB card and a
    # 9.7GB MIG slice, either of which would OOM the L95 configs and make
    # timings incomparable.
    "gpu_selector": {"nvidia.com/gpu.memory": "81920"},
    # Products the selector admits, for reporting/pinning.
    "gpu_products": ["NVIDIA-A100-SXM4-80GB", "NVIDIA-A100-80GB-PCIe"],
    "gpu_quota": 8,
    "storage_class": "rook-cephfs",      # RWX-capable; block storage is not
    "reports_pvc": "jcm-bench",
    "runs_pvc": "jcm-runs",
    "image": "ghcr.io/climate-analytics-lab/jcm:latest",
    # Leave priorityClassName UNSET. The namespace bans every named class at
    # 0 pods — including "default" — so naming one gets the pod refused by
    # quota, while an unnamed pod runs at priority 0 and is fine.
    "priority_class": None,
    # MAM4-JAX needs these and the image lacks them.
    "extra_pip": ["diffrax>=0.7", "matplotlib"],
}

# A new site must supply every key above. The ones most likely to differ:
#
#   gpu_resource    "nvidia.com/gpu" on GKE/EKS and most vanilla clusters
#   gpu_selector    GKE: {"cloud.google.com/gke-accelerator": "nvidia-a100-80gb"}
#                   EKS: {"node.kubernetes.io/instance-type": "p4d.24xlarge"}
#   storage_class   GKE: "standard-rwx" / Filestore;  EKS: "efs-sc" for RWX
#   priority_class  usually settable elsewhere; Nautilus is the odd one
#
# Add it here and pass --site <name>; do not fork the generators.
SITES = {"nautilus": NAUTILUS}


def get(name: str = "nautilus") -> dict:
    if name not in SITES:
        raise SystemExit(
            f"unknown site {name!r}; known: {', '.join(sorted(SITES))}. "
            "Add a profile to site.py rather than editing the generators.")
    return SITES[name]
