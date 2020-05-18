#include "pathperceptron.hh"

//#include <iostream>
#include <fstream>
#include "base/bitfield.hh"
#include "base/intmath.hh"
#include "debug/PAthPerceptron.hh"

//#define DEBUG
//#define RECORD

PathPerceptron::PathPerceptron(const PathPerceptronParams *params)
  : BPredUnit(params),
        globalPredictorSize(params->globalPredictorSize),
        hislen(params->hislen),
        maxhislen(params->maxhislen),
        threadHistory(params->numThreads)
{

  for (auto& history : threadHistory){
      history.R.assign(hislen+1, 0);
      history.SR.assign(hislen+1, 0);
      history.path.assign(maxhislen+1, 0);
      history.Spath.assign(maxhislen+1, 0);

      history.globalHistory = new uint8_t[maxhislen+1];
      history.sglobalHistory = new uint8_t[maxhislen+1];
      history.gHist = history.globalHistory;
      history.sgHist = history.sglobalHistory;
      memset(history.globalHistory, 0, maxhislen+1);
      memset(history.sglobalHistory, 0, maxhislen+1);
      history.ptGhist = 0;
      history.sptGhist = 0;
  }

  W.assign(globalPredictorSize, std::vector<int>\
          (hislen+1, 0));
  theta = 2.14 * hislen + 22.8;
  unsigned bitsPerWeight = 2 + ceilLog2(theta);
  maxWeight = (1 << (bitsPerWeight - 1)) - 1;
  minWeight = -(1 << (bitsPerWeight - 1));

  DPRINTFR(PAthPerceptron, "globalPredictorSize is %d, hislen is %d,\
maxhislen is %d, maxWeight is %d, minWeight is %d, theta is %d\n",
          globalPredictorSize, hislen, maxhislen, maxWeight, minWeight, theta);
}

void
PathPerceptron::btbUpdate(ThreadID tid, Addr BrAddr, void * &bp_history)
{
#ifdef RECORD
    static long count = 0;
    count++;
    DPRINTFR(PAthPerceptron,
            "[btbUpdate No.%d] BTB misss resets prediction: %lx\n",
            count, BrAddr);
#endif
}

inline int
PathPerceptron::getIndex(Addr BrAddr)
{
    return (BrAddr >> 2) % globalPredictorSize;
}

void
PathPerceptron::updateGHist(uint8_t *&h, bool taken, uint8_t *tab, int &pt)
{
    if (pt == 0) {
        for (int i = 0; i < hislen; i++)
            tab[maxhislen - hislen + i] = tab[i];
        pt = maxhislen - hislen;
        h = &tab[pt];
    }
    pt--;
    h--;
    h[0] = (taken) ? 1 : 0;
}

inline void
PathPerceptron::updatePath(std::vector<Addr> &path, Addr BrAddr)
{
    for (int i = maxhislen; i > 0; i--) {
        path[i] = path[i-1];
    }
    path[0] = BrAddr;


#ifdef DEBUG
    DPRINTFR(PAthPerceptron, "[---updatePath] Updated path with address %lx\n",
            BrAddr);
    for (int i = 0; i < hislen+1; i++) {
        DPRINTFR(PAthPerceptron, "path[%d]=0x%lx, ", i, path[i]);
    }
    DPRINTFR(PAthPerceptron, "\n");
#endif
}

inline void
PathPerceptron::updateR(std::vector<int> &R, int index, bool taken)
{
    for (int j = 1; j <= hislen; j++) {
        int k = hislen - j;
        R[k+1] = taken ? R[k] + W[index][j] : R[k] - W[index][j];
    }
    R[0] = 0;


#ifdef DEBUG
    DPRINTFR(PAthPerceptron, "[---updateR] Updated R: ");
    for (int i = 0; i < hislen+1; i++) {
        DPRINTFR(PAthPerceptron, "R[%d]=%d, ", i, R[i]);
    }
    DPRINTFR(PAthPerceptron, "\n");
#endif
}

void
PathPerceptron::updateHist(ThreadID tid, Addr BrAddr, bool taken)
{
#ifdef RECORD
    DPRINTFR(PAthPerceptron,
            "[updateHist] Calling updateHist of branch %lx %s...\n",
            BrAddr, squashed ? "during squash" : "");
#endif


    ThreadHistory &tHist = threadHistory[tid];

    updateR(tHist.SR, getIndex(BrAddr), taken);
    updatePath(tHist.Spath, BrAddr);
    updateGHist(tHist.sgHist, taken, tHist.sglobalHistory, tHist.sptGhist);
}

inline void
PathPerceptron::satInc(int &weight, bool taken)
{
    if (taken) {
        if (weight < maxWeight) {
            weight += 1;
        }
    }
    else {
        if (weight > minWeight) {
            weight -= 1;
        }
    }
}

bool
PathPerceptron::lookup(ThreadID tid, Addr BrAddr, void * &bp_history)
{
    // Get thread history
    ThreadHistory &tHist = threadHistory[tid];

    // Get weight index and make prediction
    int index = getIndex(BrAddr);
    assert(index < globalPredictorSize);

    int out = tHist.SR[hislen] + W[index][0];
    bool taken = (out >= 0);

#ifdef RECORD
    static long count = 0;
    count++;
    DPRINTFR(PAthPerceptron,
            "[lookup No.%d] Looking up branch 0x%lx on index %d: %s\n",
            count, BrAddr, index, taken ? "taken" : "not taken");
#endif

    // Save non-speculative info into branch-related history
    BPHistory *history = new BPHistory(
            BrAddr, tHist.Spath, taken, false, out, hislen, tHist.sgHist);

    bp_history = (void *) history;

    updateHist(tid, BrAddr, taken);

    return taken;
}

void
PathPerceptron::uncondBranch(ThreadID tid, Addr BrAddr, void * &bp_history)
{
#ifdef RECORD
    static long count = 0;
    count++;
    DPRINTFR(PAthPerceptron,
            "[uncond No.%d] Unconditional Branch: %lx\n", count, BrAddr);
#endif

    ThreadHistory &tHist = threadHistory[tid];

    BPHistory *history = new BPHistory(
            BrAddr, tHist.Spath, true, true, 999999, hislen, tHist.sgHist);

    bp_history = (void *) history;

    updateHist(tid, BrAddr, true);
}

void
PathPerceptron::update(ThreadID tid, Addr BrAddr, bool taken,
                        void *bp_history, bool squashed,
                        const StaticInstPtr &inst, Addr addr)
{
#ifdef RECORD
    static long count = 0;
    // Only record real updates
    if (!squashed)
        count++;
    DPRINTFR(PAthPerceptron,
            "[update No.%d] Calling update of branch %lx%s\n",
            count, BrAddr, squashed ? " during squash" : "");
#endif
    BPHistory *history = static_cast <BPHistory *>(bp_history);
    ThreadHistory &tHist = threadHistory[tid];

    // If called during squash, update hist with correct result and return
    if (squashed){
        history->isSquashed = true;
        return;
    }

    updateR(tHist.R, getIndex(BrAddr), taken);
    updatePath(tHist.path, BrAddr);
    updateGHist(tHist.gHist, taken, tHist.globalHistory, tHist.ptGhist);

    if (history->isSquashed) {
        tHist.SR = tHist.R;
        tHist.Spath = tHist.path;
        memcpy(tHist.sglobalHistory, tHist.globalHistory, maxhislen+1);
        tHist.sptGhist = tHist.ptGhist;
        tHist.sgHist = &(tHist.sglobalHistory[tHist.sptGhist]);
    }



    // Do not update on unconditional branches
    if (history->isUncond) {
        delete history;
        return;
    }

    if (history->taken != taken || abs(history->out) <= theta) {
        int i = getIndex(BrAddr);
#ifdef DEBUG
        int val_prior = W[i][0];
#endif

        satInc(W[i][0], taken);
#ifdef DEBUG
        DPRINTFR(PAthPerceptron,
                "W[%3d][0]: %4d -> %4d\n", i, val_prior, W[i][0]);
#endif

        for (int j = 1; j <= hislen; j++) {
            // Do not update on empty branch
            if (history->path[j-1] == 0x0)
                continue;
            // Get the row number
            int k = getIndex(history->path[j-1]);
#ifdef DEBUG
            int val_prior = W[k][j];
#endif
            // Saturated update

            satInc(W[k][j], taken == (history->ghist)[j-1]);
#ifdef DEBUG
            DPRINTFR(PAthPerceptron,
                    "W[%3d][%3d]: %4d -> %4d\n", k, j, val_prior,  W[k][j]);
#endif
        }
    }

    if (!squashed) delete history;
}

void
PathPerceptron::squash(ThreadID tid, void * bp_history)
{
    BPHistory *h = static_cast<BPHistory *> (bp_history);
#ifdef RECORD
    DPRINTFR(PAthPerceptron,
            "[squash] Calling squash of branch 0x%lx\n", h->BrAddr);
#endif
    delete h;
}


unsigned PathPerceptron::getGHR(ThreadID tid, void *bp_history) const
{
  BPHistory *h = static_cast<BPHistory *>(bp_history);
  unsigned val = 0;
  for (int i = 0; i < 32; i++)
      val |= ((h->ghist)[i] & 0x1) << i;
#ifdef DEBUG
  static long count = 0;
  count++;
  DPRINTFR(PAthPerceptron,
          "[getGHR No.%d] Calling getGHR and got val %x\n", count, val);
#endif
  return val;
}

PathPerceptron*
PathPerceptronParams::create()
{
  return new PathPerceptron(this);
}
