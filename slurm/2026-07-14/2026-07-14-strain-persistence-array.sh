#!/bin/bash

set -e

cd "$(dirname "$0")"

echo "configuration ==========================================================="
JOBDATE="$(date '+%Y-%m-%d')"
echo "JOBDATE ${JOBDATE}"

JOBNAME="$(basename -s .sh "$0")"
echo "JOBNAME ${JOBNAME}"

JOBPROJECT="$(basename -s .git "$(git remote get-url origin)")"
echo "JOBPROJECT ${JOBPROJECT}"

NOTEBOOK_NAME="2026-07-14-strain-persistence"
echo "NOTEBOOK_NAME ${NOTEBOOK_NAME}"
NOTEBOOK_PATH="bindle/${NOTEBOOK_NAME}.py"
echo "NOTEBOOK_PATH ${NOTEBOOK_PATH}"

# Community-assembly sweep over the 3-site model (N_SITES=3, so 2 ** 3 =
# 8 strains, genomes 0..7). Each array task t in [0, 255] initializes one
# community whose seeded strains are read off the binary code of t: bit j
# of t set <=> strain j (genome integer j) is seeded. Enumerating t over
# [0, 255] walks every subset of the 8 strains exactly once (256
# communities). The seeded strains are mixed equally --- SEED_COUNT_PER_-
# STRAIN hosts per strain --- and each community is run forward WITHOUT
# mutation (the notebook fixes MUTATION_RATE=0), so the only genomes that
# ever circulate are the seeded ones.
#
# One replicate per array task (SEED=1); there are exactly 256 array
# tasks, well under the cluster's 1000-task cap, so --- unlike the packed
# mutation sweeps --- there is no per-task CHUNK concurrency here. Each
# replicate emits a single "strainlast" parquet of 8 rows (one per
# strain) recording the last update at which each strain was observed
# (-1 = strain never added to this community; N_STEPS = strain added and
# never went extinct; 0 <= u < N_STEPS = strain added but went extinct at
# update u). Concatenating across the array yields 256 * 8 = 2048 rows.
N_SITES=3
N_STRAINS=$((1 << N_SITES))
N_ARRAY_TASKS=256
SEED=1
N_STEPS=25000
POP_SIZE=1000000
SEED_COUNT_PER_STRAIN=10
echo "N_SITES=${N_SITES} N_STRAINS=${N_STRAINS} N_ARRAY_TASKS=${N_ARRAY_TASKS}"
echo "SEED=${SEED} N_STEPS=${N_STEPS} POP_SIZE=${POP_SIZE}"
echo "SEED_COUNT_PER_STRAIN=${SEED_COUNT_PER_STRAIN}"

SOURCE_REVISION="$(git rev-parse HEAD)"
echo "SOURCE_REVISION ${SOURCE_REVISION}"
SOURCE_REMOTE_URL="$(git config --get remote.origin.url)"
echo "SOURCE_REMOTE_URL ${SOURCE_REMOTE_URL}"

echo "initialization telemetry ==============================================="
echo "date $(date)"
echo "hostname $(hostname)"
echo "PWD ${PWD}"
echo "SLURM_JOB_ID ${SLURM_JOB_ID:-nojid}"
echo "SLURM_ARRAY_TASK_ID ${SLURM_ARRAY_TASK_ID:-notid}"
module purge || :
module load Python/3.10.8 || :
echo "python3.10 $(which python3.10)"
echo "python3.10 --version $(python3.10 --version)"

echo "setup HOME dirs ========================================================"
mkdir -p "${HOME}/joblatest"
mkdir -p "${HOME}/joblog"
mkdir -p "${HOME}/jobscript"
if ! [ -e "${HOME}/scratch" ]; then
    if [ -e "/mnt/scratch/${USER}" ]; then
        ln -s "/mnt/scratch/${USER}" "${HOME}/scratch" || :
    else
        mkdir -p "${HOME}/scratch" || :
    fi
fi

echo "setup BATCHDIR =========================================================="
BATCHDIR="${HOME}/scratch/${JOBPROJECT}/${JOBNAME}/${JOBDATE}"
if [ -e "${BATCHDIR}" ]; then
    echo "BATCHDIR ${BATCHDIR} exists, clearing it"
fi
rm -rf "${BATCHDIR}"
mkdir -p "${BATCHDIR}"
echo "BATCHDIR ${BATCHDIR}"

echo "symlinking latest"
LATESTDIR="${HOME}/scratch/${JOBPROJECT}/${JOBNAME}/latest"
echo "${BATCHDIR} > ${LATESTDIR}"
ln -sfn "${BATCHDIR}" "${LATESTDIR}"

BATCHDIR_JOBLOG="${BATCHDIR}/joblog"
echo "BATCHDIR_JOBLOG ${BATCHDIR_JOBLOG}"
mkdir -p "${BATCHDIR_JOBLOG}"

BATCHDIR_JOBRESULT="${BATCHDIR}/jobresult"
echo "BATCHDIR_JOBRESULT ${BATCHDIR_JOBRESULT}"
mkdir -p "${BATCHDIR_JOBRESULT}"

BATCHDIR_JOBSCRIPT="${BATCHDIR}/jobscript"
echo "BATCHDIR_JOBSCRIPT ${BATCHDIR_JOBSCRIPT}"
mkdir -p "${BATCHDIR_JOBSCRIPT}"

BATCHDIR_JOBSOURCE="${BATCHDIR}/_jobsource"
echo "BATCHDIR_JOBSOURCE ${BATCHDIR_JOBSOURCE}"
if [[ $* == *--dirty* ]]; then
    cp -r "$(git rev-parse --show-toplevel)" "${BATCHDIR_JOBSOURCE}"
else
    mkdir -p "${BATCHDIR_JOBSOURCE}"
    for attempt in {1..5}; do
        rm -rf "${BATCHDIR_JOBSOURCE}/.git"
        git -C "${BATCHDIR_JOBSOURCE}" init \
        && git -C "${BATCHDIR_JOBSOURCE}" remote add origin "${SOURCE_REMOTE_URL}" \
        && git -C "${BATCHDIR_JOBSOURCE}" fetch origin "${SOURCE_REVISION}" --depth=1 \
        && git -C "${BATCHDIR_JOBSOURCE}" reset --hard FETCH_HEAD \
        && break || echo "failed to clone, retrying..."
        if [ $attempt -eq 5 ]; then
            echo "failed to clone, failing"
            exit 1
        fi
        sleep 5
    done
fi

BATCHDIR_ENV="${BATCHDIR}/_jobenv"
python3.10 -m venv --system-site-packages "${BATCHDIR_ENV}"
source "${BATCHDIR_ENV}/bin/activate"
echo "python3.10 $(which python3.10)"
echo "python3.10 --version $(python3.10 --version)"
for attempt in {1..5}; do
    python3.10 -m pip install --upgrade pip 'setuptools<75' wheel || :
    python3.10 -m pip install --upgrade uv \
    && python3.10 -m uv pip install joinem==0.11.1 \
    && python3.10 -m uv pip install \
        -r "${BATCHDIR_JOBSOURCE}/requirements.txt" \
    && break || echo "pip install attempt ${attempt} failed"
    if [ ${attempt} -eq 5 ]; then
        echo "pip install failed"
        exit 1
    fi
done

echo "setup dependencies ========================================== \${SECONDS}"
source "${BATCHDIR_ENV}/bin/activate"
python3.10 -m uv pip freeze

echo "sbatch preamble ========================================================="
JOB_PREAMBLE=$(cat << EOF
set -e
shopt -s globstar

# adapted from https://unix.stackexchange.com/a/504829
handlefail() {
    echo ">>>error<<<" || :
    awk 'NR>L-4 && NR<L+4 { printf "%-5d%3s%s\n",NR,(NR==L?">>>":""),\$0 }' L=\$1 \$0 || :
    ln -sfn "\${JOBSCRIPT}" "\${HOME}/joblatest/jobscript.failed" || :
    ln -sfn "\${JOBLOG}" "\${HOME}/joblatest/joblog.failed" || :
    $(which scontrol || which echo) requeuehold "${SLURM_JOBID:-nojid}"
}
trap 'handlefail $LINENO' ERR

echo "initialization telemetry ------------------------------------ \${SECONDS}"
echo "SOURCE_REVISION ${SOURCE_REVISION}"
echo "BATCHDIR ${BATCHDIR}"

echo "cc SLURM script --------------------------------------------- \${SECONDS}"
JOBSCRIPT="\${HOME}/jobscript/\${SLURM_JOB_ID:-nojid}"
echo "JOBSCRIPT \${JOBSCRIPT}"
cp "\${0}" "\${JOBSCRIPT}"
chmod +x "\${JOBSCRIPT}"
cp "\${JOBSCRIPT}" "${BATCHDIR_JOBSCRIPT}/\${SLURM_JOB_ID:-nojid}"
ln -sfn "\${JOBSCRIPT}" "${HOME}/joblatest/jobscript.launched"

echo "cc job log -------------------------------------------------- \${SECONDS}"
JOBLOG="\${HOME}/joblog/\${SLURM_JOB_ID:-nojid}"
echo "JOBLOG \${JOBLOG}"
touch "\${JOBLOG}"
ln -sfn "\${JOBLOG}" "${BATCHDIR_JOBLOG}/\${SLURM_JOB_ID:-nojid}"
ln -sfn "\${JOBLOG}" "\${HOME}/joblatest/joblog.launched"

echo "setup JOBDIR ------------------------------------------------ \${SECONDS}"
JOBDIR="${BATCHDIR}/__\${SLURM_ARRAY_TASK_ID:-\${SLURM_JOB_ID:-\${RANDOM}}}"
echo "JOBDIR \${JOBDIR}"
if [ -e "\${JOBDIR}" ]; then
    echo "JOBDIR \${JOBDIR} exists, clearing it"
fi
rm -rf "\${JOBDIR}"
mkdir -p "\${JOBDIR}"
cd "\${JOBDIR}"
echo "PWD \${PWD}"

echo "job telemetry ----------------------------------------------- \${SECONDS}"
echo "source SLURM_JOB_ID ${SLURM_JOB_ID:-nojid}"
echo "current SLURM_JOB_ID \${SLURM_JOB_ID:-nojid}"
echo "SLURM_ARRAY_TASK_ID \${SLURM_ARRAY_TASK_ID:-notid}"
echo "hostname \$(hostname)"
echo "date \$(date)"

echo "module setup ------------------------------------------------ \${SECONDS}"
module purge || :
module load Python/3.10.8 || :
echo "python3.10 \$(which python3.10)"
echo "python3.10 --version \$(python3.10 --version)"

echo "setup dependencies- ----------------------------------------- \${SECONDS}"
source "${BATCHDIR_ENV}/bin/activate"
python3.10 -m uv pip freeze

EOF
)

echo "create sbatch file: work ==============================================="

SBATCH_FILE="$(mktemp)"
echo "SBATCH_FILE ${SBATCH_FILE}"

###############################################################################
# WORK ---------------------------------------------------------------------- #
###############################################################################
cat > "${SBATCH_FILE}" << EOF
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output="/mnt/home/%u/joblog/%j"
#SBATCH --mail-user=mawni4ah2o@pomail.net
#SBATCH --mail-type=FAIL,TIME_LIMIT,ARRAY_TASKS
#SBATCH --account=ecode
#SBATCH --requeue
#SBATCH --array=0-$((N_ARRAY_TASKS - 1))

${JOB_PREAMBLE}

echo "lscpu ------------------------------------------------------- \${SECONDS}"
lscpu || :

echo "cpuinfo ----------------------------------------------------- \${SECONDS}"
cat /proc/cpuinfo || :

echo "marimo notebook source -------------------------------------- \${SECONDS}"
echo "notebook source: ${BATCHDIR_JOBSOURCE}/${NOTEBOOK_PATH}"
cat "${BATCHDIR_JOBSOURCE}/${NOTEBOOK_PATH}" || :

echo "task assignment --------------------------------------------- \${SECONDS}"
# The array task id IS the community index (array_id) --- its 8-bit
# binary code selects which of the 8 strains are seeded (see notebook).
ARRAY_ID=\${SLURM_ARRAY_TASK_ID:-0}
echo "ARRAY_ID=\${ARRAY_ID} N_SITES=${N_SITES} SEED=${SEED}"
printf 'community bits: %08d\n' "\$(echo "obase=2; \${ARRAY_ID}" | bc)" || :

# Single-threaded numpy so the one replicate stays within its one CPU
# (--cpus-per-task=1).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "do work ----------------------------------------------------- \${SECONDS}"
# Run one community replicate on CPU (engine=numpy). The notebook writes
# its single "strainlast" parquet to
# outdata/${NOTEBOOK_NAME}/strainlast/<uid>/... with the simulation
# parameters, the array_id, and a uuid replicate identifier stamped onto
# every row, so the community is fully recoverable downstream.
#
# Export from a PRIVATE per-task copy of the notebook. Every array task
# would otherwise run marimo against the same notebook file inside the
# single _jobsource clone on the network filesystem; under that
# concurrency marimo can clobber the shared source to an empty default
# stub, after which later exports yield a blank notebook with no outdata.
# A private copy removes that shared-file race. The notebook does
# 'from pylib import ...', and marimo puts the notebook's own directory
# on sys.path, so the private copy sits beside a pylib symlink.
NBDIR="\${JOBDIR}/_nb"
mkdir -p "\${NBDIR}"
cp "${BATCHDIR_JOBSOURCE}/${NOTEBOOK_PATH}" "\${NBDIR}/${NOTEBOOK_NAME}.py"
ln -sfn "${BATCHDIR_JOBSOURCE}/pylib" "\${NBDIR}/pylib"

MPLBACKEND=Agg python3.10 -m marimo export ipynb \
    --include-outputs --sort topological -f \
    "\${NBDIR}/${NOTEBOOK_NAME}.py" \
    -o "\${JOBDIR}/${NOTEBOOK_NAME}.ipynb" \
    -- \
    --array-id "\${ARRAY_ID}" \
    --seed ${SEED} \
    --n-steps ${N_STEPS} \
    --pop-size ${POP_SIZE} \
    --seed-count-per-strain ${SEED_COUNT_PER_STRAIN} \
    --engine numpy \
    --skip-plotting True

# Fail loudly on a blank/failed export. marimo can exit 0 while producing
# a notebook whose cells never executed --- and the run cell is what
# writes the parquet --- so a "successful" export with no outputs would
# otherwise sail through to >>>complete<<<. Require both a non-trivial
# exported notebook and the run cell's parquet output under outdata/<nb>/.
NB_OUT="\${JOBDIR}/${NOTEBOOK_NAME}.ipynb"
OUTDATA_DIR="\${JOBDIR}/outdata/${NOTEBOOK_NAME}"
NB_BYTES=\$(wc -c < "\${NB_OUT}" 2>/dev/null || echo 0)
if [ "\${NB_BYTES}" -lt 10000 ]; then
    echo "ERROR [array_id=\${ARRAY_ID}]: exported notebook \${NB_OUT} missing or trivial (\${NB_BYTES} bytes)"
    exit 1
fi
N_PQT=\$(find "\${OUTDATA_DIR}" -name 'a=*+ext=.pqt' 2>/dev/null | wc -l)
if [ "\${N_PQT}" -lt 1 ]; then
    echo "ERROR [array_id=\${ARRAY_ID}]: no parquet outputs under \${OUTDATA_DIR} (notebook produced no outdata)"
    exit 1
fi
echo "  [array_id=\${ARRAY_ID}] export OK: \${NB_BYTES} byte notebook, \${N_PQT} parquet(s) in outdata"

echo "finalization telemetry -------------------------------------- \${SECONDS}"
ls -lR "\${JOBDIR}" | head -200
du -sh "\${JOBDIR}"
ln -sfn "\${JOBSCRIPT}" "${HOME}/joblatest/jobscript.finished"
ln -sfn "\${JOBLOG}" "${HOME}/joblatest/joblog.finished"
echo "SECONDS \${SECONDS}"
echo '>>>complete<<<'

EOF
###############################################################################
# --------------------------------------------------------------------------- #
###############################################################################


echo "submit sbatch file ====================================================="
$(which sbatch && echo --job-name="${JOBNAME}" || which bash) "${SBATCH_FILE}"

echo "create sbatch file: collate ============================================"

SBATCH_FILE="$(mktemp)"
echo "SBATCH_FILE ${SBATCH_FILE}"

###############################################################################
# COLLATE ------------------------------------------------------------------- #
###############################################################################
cat > "${SBATCH_FILE}" << EOF
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output="/mnt/home/%u/joblog/%j"
#SBATCH --mail-user=mawni4ah2o@pomail.net
#SBATCH --mail-type=ALL
#SBATCH --account=ecode
#SBATCH --requeue

${JOB_PREAMBLE}

echo "BATCHDIR ${BATCHDIR}"
ls -l "${BATCHDIR}"

echo "finalize ---------------------------------------------------- \${SECONDS}"
echo "   - archive job dir"
pushd "${BATCHDIR}/.."
    tar czf \
    "${BATCHDIR_JOBRESULT}/a=jobarchive+date=${JOBDATE}+job=${JOBNAME}+ext=.tar.gz" \
    "\$(basename "${BATCHDIR}")"/__*
popd

echo "   - join per-replicate parquets across all communities"
# Each per-replicate parquet already carries the replicate_uid, the
# array_id, and the condition columns (n_sites, pop_size, n_steps, seed,
# engine, mutation_rate, ...) stamped by the notebook's run cell, so a
# straight concatenation yields a self-describing collated frame. The
# strainlast parquet is the sole output --- 8 rows per replicate (one per
# strain) recording each strain's last-observed update. 256 communities
# x 8 strains = 2048 rows in the joined frame.
for kind in strainlast; do
    echo "    joining \${kind} ..."
    out_path="${BATCHDIR_JOBRESULT}/a=\${kind}+date=${JOBDATE}+job=${JOBNAME}+ext=.pqt"
    # __<arrayid>/outdata/... --- one output dir per array task (globstar
    # ** also tolerates a nested layout).
    ls -1 "${BATCHDIR}"/__*/**/outdata/${NOTEBOOK_NAME}/\${kind}/*/a=\${kind}+*+ext=.pqt 2>/dev/null \
        | tee /dev/stderr \
        | python3.10 -m joinem --progress "\${out_path}" \
        || echo "no \${kind} files to join"
done
ls -l "${BATCHDIR_JOBRESULT}"
du -h "${BATCHDIR_JOBRESULT}"

echo "   - archive joblog"
pushd "${BATCHDIR}"
    tar czf \
    "${BATCHDIR_JOBRESULT}/a=joblog+date=${JOBDATE}+job=${JOBNAME}+ext=.tar.gz" \
    -h "\$(basename "${BATCHDIR_JOBLOG}")"
popd

echo "   - archive jobscript"
pushd "${BATCHDIR}"
    tar czf \
    "\$(basename "${BATCHDIR_JOBRESULT}")/a=jobscript+date=${JOBDATE}+job=${JOBNAME}+ext=.tar.gz" \
    -h "\$(basename ${BATCHDIR_JOBSCRIPT})"
popd

ls -l "${BATCHDIR}"

echo "cleanup ----------------------------------------------------- \${SECONDS}"
cd "${BATCHDIR}"
for f in _*; do
    echo "tar and rm \$f"
    tar cf "\${f}.tar" -h "\${f}"
    rm -rf "\${f}"
done
cd
ls -l "${BATCHDIR}"

echo "finalization telemetry -------------------------------------- \${SECONDS}"
ln -sfn "\${JOBSCRIPT}" "\${HOME}/joblatest/jobscript.completed"
ln -sfn "\${JOBLOG}" "\${HOME}/joblatest/joblog.completed"
ln -sfn "${BATCHDIR_JOBRESULT}" "\${HOME}/joblatest/jobresult.completed"
echo "SECONDS \${SECONDS}"
echo '>>>complete<<<'

EOF
###############################################################################
# --------------------------------------------------------------------------- #
###############################################################################

echo "submit sbatch file ====================================================="
$(which sbatch && echo --job-name="${JOBNAME}" --dependency=singleton || which bash) "${SBATCH_FILE}"

echo "finalization telemetry ================================================="
echo "BATCHDIR ${BATCHDIR}"
echo "SECONDS ${SECONDS}"
echo '>>>complete<<<'
