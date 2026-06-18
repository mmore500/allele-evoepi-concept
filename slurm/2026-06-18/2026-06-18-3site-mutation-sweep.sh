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

NOTEBOOK_NAME="2026-05-20-founder"
echo "NOTEBOOK_NAME ${NOTEBOOK_NAME}"
NOTEBOOK_PATH="bindle/${NOTEBOOK_NAME}.py"
echo "NOTEBOOK_PATH ${NOTEBOOK_PATH}"

# Sweep: the 3-site model across a geometric low-mutation-rate sweep
# with ~2 points per decade, from 3e-5 down to 1e-10, and 100
# replicates per rate.
# 12 MUTATION_RATE conditions x 100 replicates = 1200 replicates.
#
# The cluster caps a job array at 1000 queued tasks, so we pack CHUNK=4
# replicates into each array task and run those 4 *concurrently* (one
# CPU each, see --cpus-per-task below) rather than sequentially --- this
# folds 1200 replicates into ceil(1200 / 4) = 300 array tasks (under the
# 1000 cap) while keeping per-task walltime ~1x a single replicate.
#
# Global replicate index r in [0, N_TASKS): condition is
# MUTATION_RATES[r / 100], seed is (r % 100) + 1. Array task t owns the
# CHUNK consecutive indices r = t * CHUNK + j for j in [0, CHUNK), each
# launched as a background job (indices >= N_TASKS are skipped on the
# final partial chunk). N_SITES is fixed at 3 (the 3-site model).
N_SITES=3
MUTATION_RATES=(
    3e-5
    1e-5 3e-6
    1e-6 3e-7
    1e-7 3e-8
    1e-8 3e-9
    1e-9 3e-10
    1e-10
)
N_CONDITIONS=${#MUTATION_RATES[@]}
N_REPLICATES=100
N_TASKS=$((N_CONDITIONS * N_REPLICATES))
CHUNK=4
N_ARRAY_TASKS=$(((N_TASKS + CHUNK - 1) / CHUNK))
N_STEPS=25000
POP_SIZE=1000000
echo "N_SITES=${N_SITES} MUTATION_RATES=${MUTATION_RATES[*]}"
echo "N_REPLICATES=${N_REPLICATES} N_CONDITIONS=${N_CONDITIONS}"
echo "N_TASKS=${N_TASKS} CHUNK=${CHUNK} N_ARRAY_TASKS=${N_ARRAY_TASKS}"
echo "N_STEPS=${N_STEPS} POP_SIZE=${POP_SIZE}"

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
#SBATCH --cpus-per-task=${CHUNK}
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output="/mnt/home/%u/joblog/%j"
#SBATCH --mail-user=mawni4ah2o@pomail.net
#SBATCH --mail-type=FAIL,TIME_LIMIT,ARRAY_TASKS
#SBATCH --account=ecode
#SBATCH --requeue
#SBATCH --array=0-299

${JOB_PREAMBLE}

echo "lscpu ------------------------------------------------------- \${SECONDS}"
lscpu || :

echo "cpuinfo ----------------------------------------------------- \${SECONDS}"
cat /proc/cpuinfo || :

echo "marimo notebook source -------------------------------------- \${SECONDS}"
echo "notebook source: ${BATCHDIR_JOBSOURCE}/${NOTEBOOK_PATH}"
cat "${BATCHDIR_JOBSOURCE}/${NOTEBOOK_PATH}" || :

echo "task assignment --------------------------------------------- \${SECONDS}"
MUTATION_RATES=(${MUTATION_RATES[*]})
TASK_ID=\${SLURM_ARRAY_TASK_ID:-0}
echo "TASK_ID=\${TASK_ID} CHUNK=${CHUNK} N_SITES=${N_SITES}"
echo "owns global replicate indices \$((TASK_ID * ${CHUNK})) .. \$((TASK_ID * ${CHUNK} + ${CHUNK} - 1))"

# Each of the CHUNK replicates runs as its own single-threaded process
# so all CHUNK share the array task's CPUs (--cpus-per-task=${CHUNK})
# without oversubscribing: pin the numeric libraries to one thread each.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Run one founder replicate (one global index) on CPU (engine=numpy, no
# GPU requested in #SBATCH). --skip-plotting=True drops the per-replicate
# teeplots --- the parquets are the deliverable. Each replicate runs in
# its own working dir \${JOBDIR}/r<gid> and the notebook writes parquets
# to that dir's outdata/${NOTEBOOK_NAME}/<kind>/<uid>/... with simulation
# parameters (n_sites, pop_size, n_steps, seed, mutation_rate, ...) and a
# uuid replicate identifier stamped onto every row, so the mutation rate
# is recoverable downstream.
run_replicate() {
    local gid="\$1"
    local rate="\${MUTATION_RATES[\$((gid / ${N_REPLICATES}))]}"
    local seed="\$((gid % ${N_REPLICATES} + 1))"
    local repdir="\${JOBDIR}/r\${gid}"
    mkdir -p "\${repdir}"
    cd "\${repdir}"
    echo "  [gid=\${gid}] N_SITES=${N_SITES} MUTATION_RATE=\${rate} SEED=\${seed} repdir=\${repdir}"
    MPLBACKEND=Agg python3.10 -m marimo export ipynb \
        --include-outputs --sort topological -f \
        "${BATCHDIR_JOBSOURCE}/${NOTEBOOK_PATH}" \
        -o "\${repdir}/${NOTEBOOK_NAME}.ipynb" \
        -- \
        --n-sites ${N_SITES} \
        --mutation-rate "\${rate}" \
        --seed "\${seed}" \
        --n-steps ${N_STEPS} \
        --pop-size ${POP_SIZE} \
        --engine numpy \
        --skip-plotting True

    # Fail loudly on a blank/failed export. marimo can exit 0 while
    # producing a notebook whose cells never executed --- and the run
    # cell is what writes the parquets --- so a "successful" export with
    # no outputs would otherwise sail through to >>>complete<<<. Require
    # both a non-trivial exported notebook and the run cell's parquet
    # outputs under outdata/<nb>/, else fail the replicate (return 1,
    # caught by the wait loop below).
    local nb_out="\${repdir}/${NOTEBOOK_NAME}.ipynb"
    local outdata_dir="\${repdir}/outdata/${NOTEBOOK_NAME}"
    local nb_bytes
    nb_bytes=\$(wc -c < "\${nb_out}" 2>/dev/null || echo 0)
    if [ "\${nb_bytes}" -lt 10000 ]; then
        echo "ERROR [gid=\${gid}]: exported notebook \${nb_out} missing or trivial (\${nb_bytes} bytes)"
        return 1
    fi
    local n_pqt
    n_pqt=\$(find "\${outdata_dir}" -name 'a=*+ext=.pqt' 2>/dev/null | wc -l)
    if [ "\${n_pqt}" -lt 1 ]; then
        echo "ERROR [gid=\${gid}]: no parquet outputs under \${outdata_dir} (notebook produced no outdata)"
        return 1
    fi
    echo "  [gid=\${gid}] export OK: \${nb_bytes} byte notebook, \${n_pqt} parquet(s) in outdata"
}

echo "do work (CHUNK=${CHUNK} replicates concurrently) ------------ \${SECONDS}"
declare -A REP_PID
for j in \$(seq 0 \$((${CHUNK} - 1))); do
    GID=\$((TASK_ID * ${CHUNK} + j))
    if [ "\${GID}" -ge ${N_TASKS} ]; then
        echo "  skipping global index \${GID} (>= ${N_TASKS}, partial final chunk)"
        continue
    fi
    run_replicate "\${GID}" &
    REP_PID[\${GID}]=\$!
done

echo "launched \${#REP_PID[@]} replicate(s); waiting -------------- \${SECONDS}"
WORK_FAIL=0
for gid in "\${!REP_PID[@]}"; do
    if wait "\${REP_PID[\${gid}]}"; then
        echo "  replicate gid=\${gid} ok"
    else
        echo "  replicate gid=\${gid} FAILED (pid \${REP_PID[\${gid}]})"
        WORK_FAIL=1
    fi
done
if [ "\${WORK_FAIL}" -ne 0 ]; then
    echo "one or more replicates failed; failing array task"
    exit 1
fi

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

echo "   - join per-replicate parquets across all conditions"
# Each per-replicate parquet already carries the replicate_uid plus the
# condition columns (n_sites, pop_size, n_steps, seed, engine, pow,
# mutation_rate, contact_rate, ...) stamped by the notebook's run cell,
# so a straight concatenation yields a self-describing collated frame
# that includes the swept mutation_rate. The strain parquet is the
# headline output --- one row per (replicate_uid, Step, strain)
# recording the per-genome infection count (n_cases) and new-infection
# count (n_new_infections) alongside the population-fraction
# counterparts. The hw parquet aggregates those per-genome counts by
# Hamming weight (n_cases per HW band).
for kind in strain hw traj records phylo; do
    echo "    joining \${kind} ..."
    out_path="${BATCHDIR_JOBRESULT}/a=\${kind}+date=${JOBDATE}+job=${JOBNAME}+ext=.pqt"
    # __<arrayid>/r<gid>/outdata/... --- one r<gid> subdir per packed
    # replicate (globstar ** also tolerates the un-nested layout).
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
