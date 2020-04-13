#ifndef _PATHPERCEPTRON_
#define _PATHPERCEPTRON_

#include <cstdlib>
#include <vector>
#include <fstream>
#include <cstdio>

#include "base/types.hh"
#include "cpu/pred/bpred_unit.hh"
#include "cpu/pred/sat_counter.hh"
#include "params/PathPerceptron.hh"

class PathPerceptron : public BPredUnit
{
public:

  PathPerceptron(const PathPerceptronParams *params);
  bool lookup(ThreadID tid, Addr branch_addr, void * &bp_history);
  void uncondBranch(ThreadID tid, Addr pc, void * &bp_history);
  void btbUpdate(ThreadID tid, Addr branch_addr, void * &bp_history);
  void update(ThreadID tid, Addr branch_addr, bool taken, \
          void *bp_history, bool squashed);
  void squash(ThreadID tid, void *bp_history);
  unsigned getGHR(ThreadID tid, void *bp_history) const;

private:

  std::vector<std::vector<int>> W;

  unsigned globalPredictorSize;
  int maxWeight;
  int minWeight;
  unsigned theta;
  unsigned hislen;
  unsigned maxhislen;
  unsigned bitsPerWeight;
  
  inline int getIndex(Addr BrAddr);
  inline void updatePath(std::vector<Addr> &path, Addr BrAddr);
  inline void updateR(std::vector<int> &SR, int index, bool taken);
  inline void satInc(int &weight, bool taken);
  void updateHist(ThreadID tid, Addr BrAddr, bool taken);
  void updateGHist(uint8_t *&h, bool taken, uint8_t *tab, int &pt);

  struct ThreadHistory{
      std::vector<int> R;
      std::vector<int> SR;
      std::vector<Addr> path;
      std::vector<Addr> Spath;

      uint8_t *gHist; // GlobalHistory base
      uint8_t *globalHistory;
      uint8_t *sglobalHistory;
      uint8_t *sgHist; // Speculative global history pointer
      int ptGhist;   // Non-speculative index
      int sptGhist;
  };
  std::vector<ThreadHistory> threadHistory;


  struct BPHistory {
        std::vector<Addr> path; // Path of index
        Addr BrAddr;
        bool taken;
        bool isUncond;
        bool isSquashed;
        int out;

        uint8_t *ghist;

        BPHistory(Addr pc, std::vector<Addr> path_in,
                bool taken_in, bool isUncond_in, int out_in,
                int hislen, uint8_t *gH){
            BrAddr = pc;
            path = path_in;
            taken = taken_in;
            isUncond = isUncond_in;
            out = out_in;
            isSquashed = false;
            ghist = new uint8_t[hislen];
            memcpy(ghist, gH, hislen);
        }
        ~BPHistory(){
            delete[] ghist;
        }
  };
};

#endif
