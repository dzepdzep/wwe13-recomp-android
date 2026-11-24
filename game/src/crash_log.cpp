// wwe13 - crash logs: when the game dies (a fatal error ends it with abort(), or a fault outside the recompiled
// code), crash_logs/crash-<date>-<time>.txt is written next to logs/: why it ended (signal, fault address, thread)
// and the last lines of the run's log (the [FATAL] line is among them). The launcher shows the newest one and can
// save it to Downloads. At most 20 are kept.
#include <dirent.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <dlfcn.h>
#include <ucontext.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <sys/syscall.h>

#include <rex/logging.h>

extern "C" uint32_t wwe13_last_cmd[16];
extern "C" uint32_t wwe13_cmd_ring[8][6];
extern "C" uint32_t wwe13_cmd_ring_pos;
extern "C" uint8_t* wwe13_guest_base;

namespace {
struct sigaction g_old[32];
bool g_old_valid[32];
volatile sig_atomic_t g_writing = 0;
char g_dir[512];  // (the working folder: the app's files folder)

const char* SigName(int sig) {
  switch (sig) {
    case SIGABRT: return "SIGABRT (the game ended itself: see the [FATAL] line below)";
    case SIGSEGV: return "SIGSEGV (bad memory access)";
    case SIGBUS: return "SIGBUS (bad memory access)";
    case SIGILL: return "SIGILL (illegal instruction)";
    case SIGFPE: return "SIGFPE (arithmetic error)";
    default: return "signal";
  }
}

void Put(int fd, const char* s) {
  if (s) (void)!write(fd, s, strlen(s));
}

// The newest logs/wwe13_*.log.
std::string NewestLog() {
  std::string dir = std::string(g_dir) + "/logs", best;
  time_t best_t = 0;
  if (DIR* d = opendir(dir.c_str())) {
    while (dirent* e = readdir(d)) {
      if (strncmp(e->d_name, "wwe13_", 6) != 0 || !strstr(e->d_name, ".log")) continue;
      struct stat st{};
      const std::string p = dir + "/" + e->d_name;
      if (stat(p.c_str(), &st) == 0 && st.st_mtime >= best_t) best_t = st.st_mtime, best = p;
    }
    closedir(d);
  }
  return best;
}

// Keeps the 20 newest crash logs.
void Prune(const std::string& cdir) {
  std::vector<std::string> names;
  if (DIR* d = opendir(cdir.c_str())) {
    while (dirent* e = readdir(d))
      if (strncmp(e->d_name, "crash-", 6) == 0) names.push_back(e->d_name);
    closedir(d);
  }
  if (names.size() <= 20) return;
  std::sort(names.begin(), names.end());
  for (size_t i = 0; i + 20 < names.size(); ++i) unlink((cdir + "/" + names[i]).c_str());
}

// "where": the host code address and, through dladdr, the recompiled function (sub_<guest address>) it belongs to.
void Where(int fd, const char* what, uintptr_t pc) {
  char line[320];
  Dl_info di{};
  if (pc && dladdr(reinterpret_cast<void*>(pc), &di) && di.dli_sname) {
    snprintf(line, sizeof line, "%s: %p %s+0x%lx (%s)\n", what, reinterpret_cast<void*>(pc), di.dli_sname,
             (unsigned long)(pc - uintptr_t(di.dli_saddr)), di.dli_fname ? strrchr(di.dli_fname, '/') + 1 : "?");
  } else {
    snprintf(line, sizeof line, "%s: %p\n", what, reinterpret_cast<void*>(pc));
  }
  Put(fd, line);
}

void Write(int sig, siginfo_t* info, void* ucv = nullptr) {
  if (g_writing) return;
  g_writing = 1;
  if (auto l = rex::GetLogger()) l->flush();
  if (!getcwd(g_dir, sizeof g_dir)) strcpy(g_dir, ".");  // (the [FATAL] line is in the file before abort(); a fault may not be)
  const std::string cdir = std::string(g_dir) + "/crash_logs";
  mkdir(cdir.c_str(), 0775);
  char stamp[32];
  const time_t now = time(nullptr);
  struct tm tmv{};
  localtime_r(&now, &tmv);
  strftime(stamp, sizeof stamp, "%Y%m%d-%H%M%S", &tmv);
  const std::string path = cdir + "/crash-" + stamp + ".txt";
  const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0664);
  if (fd < 0) return;
  char line[256];
  snprintf(line, sizeof line, "WWE '13 crash log %s\nReason: %s\nFault address: %p, thread %d\n", stamp,
           SigName(sig), info ? info->si_addr : nullptr, int(gettid()));
  Put(fd, line);
#if defined(__aarch64__)
  if (ucv) {
    const auto* uc = static_cast<const ucontext_t*>(ucv);
    Where(fd, "Code", uintptr_t(uc->uc_mcontext.pc));
    Where(fd, "Called from", uintptr_t(uc->uc_mcontext.regs[30]));
    // (the frame-pointer chain: a few callers further up)
    uintptr_t fp = uintptr_t(uc->uc_mcontext.regs[29]);
    for (int i = 0; i < 12 && fp && !(fp & 7); ++i) {
      const uintptr_t* f = reinterpret_cast<const uintptr_t*>(fp);
      const uintptr_t next = f[0], ret = f[1];
      if (!ret) break;
      Where(fd, "  caller", ret);
      if (next <= fp) break;
      fp = next;
    }
  }
#endif
  {
    char cl[256];
    int k = snprintf(cl, sizeof cl, "Last render command at %08X (from %08X):", wwe13_last_cmd[0], wwe13_last_cmd[1]);
    for (int i = 0; i < 12 && k < 230; ++i) k += snprintf(cl + k, sizeof cl - k, " %08X", wwe13_last_cmd[2 + i]);
    snprintf(cl + k, sizeof cl - k, "\n");
    Put(fd, cl);
    // the commands before it (oldest first)
    for (uint32_t i = 0; i < 8; ++i) {
      const uint32_t* r = wwe13_cmd_ring[(wwe13_cmd_ring_pos + i) & 7];
      snprintf(cl, sizeof cl, "  cmd %08X: %08X %08X %08X %08X %08X\n", r[0], r[1], r[2], r[3], r[4], r[5]);
      Put(fd, cl);
    }
    // what the last command's pointer arguments point at (guest physical 0xA0000000-0xBFFFFFFF), 16 words each
    if (uint8_t* b = wwe13_guest_base) {
      for (int w = 3; w < 14; ++w) {
        const uint32_t a = wwe13_last_cmd[w];
        if (a < 0xA0000000u || a >= 0xC0000000u || (a & 3)) continue;
        int q = snprintf(cl, sizeof cl, "  [%08X]:", a);
        for (int j = 0; j < 16 && q < 230; ++j) {
          const uint8_t* p = b + a + 4 * j;
          q += snprintf(cl + q, sizeof cl - q, " %02X%02X%02X%02X", p[0], p[1], p[2], p[3]);
        }
        snprintf(cl + q, sizeof cl - q, "\n");
        Put(fd, cl);
      }
    }
  }
  const std::string log = NewestLog();
  Put(fd, "Log: ");
  Put(fd, log.c_str());
  Put(fd, "\n\n--- last lines of the log ---\n");
  // The last 400 lines (gpu statistics lines left out).
  if (FILE* f = fopen(log.c_str(), "rb")) {
    fseek(f, 0, SEEK_END);
    const long size = ftell(f), from = size > 600000 ? size - 600000 : 0;
    fseek(f, from, SEEK_SET);
    std::string buf(size_t(size - from), '\0');
    buf.resize(fread(buf.data(), 1, buf.size(), f));
    fclose(f);
    std::vector<std::string> keep;
    size_t p = 0;
    while (p < buf.size()) {
      size_t q = buf.find('\n', p);
      if (q == std::string::npos) q = buf.size();
      std::string l = buf.substr(p, q - p);
      p = q + 1;
      if (l.find("] [gpu] ") != std::string::npos || l.find("[native]") != std::string::npos) continue;
      keep.push_back(std::move(l));
      if (keep.size() > 400) keep.erase(keep.begin());
    }
    for (const auto& l : keep) {
      Put(fd, l.c_str());
      Put(fd, "\n");
    }
  }
  close(fd);
  Prune(cdir);
}

void Handler(int sig, siginfo_t* info, void* uc) {
  Write(sig, info, uc);
  // On to the handler installed before (Android's debuggerd, which ends the process), else the default action.
  if (sig > 0 && sig < 32 && g_old_valid[sig]) {
    sigaction(sig, &g_old[sig], nullptr);
  } else {
    signal(sig, SIG_DFL);
  }
  if (!info || info->si_code <= 0 || sig == SIGABRT) raise(sig);
}


// Stack probe (testing): while the file <files>/stackprobe exists, every busy thread (more than 50 ms of CPU in the
// last second) gets SIGUSR2 once a second, and its handler appends the host code address and frame-pointer chain
// (recompiled function names) to <files>/stackprobe.txt. Deleting stackprobe stops it. Finds where a frozen guest
// thread spins.
void ProbeHandler(int, siginfo_t*, void* ucv) {
#if defined(__aarch64__)
  const int fd = open("stackprobe.txt", O_WRONLY | O_CREAT | O_APPEND, 0664);
  if (fd < 0) return;
  char line[96];
  char name[32] = {};
  {
    const int nf = open("/proc/thread-self/comm", O_RDONLY);
    if (nf >= 0) { (void)!read(nf, name, sizeof name - 1); close(nf); }
    for (char* c = name; *c; ++c) if (*c == '\n') *c = 0;
  }
  snprintf(line, sizeof line, "--- thread %d %s\n", int(gettid()), name);
  Put(fd, line);
  const auto* uc = static_cast<const ucontext_t*>(ucv);
  Where(fd, "pc", uintptr_t(uc->uc_mcontext.pc));
  Where(fd, "lr", uintptr_t(uc->uc_mcontext.regs[30]));
  uintptr_t fp = uintptr_t(uc->uc_mcontext.regs[29]);
  for (int i = 0; i < 16 && fp && !(fp & 7); ++i) {
    const uintptr_t* f = reinterpret_cast<const uintptr_t*>(fp);
    const uintptr_t next = f[0], ret = f[1];
    if (!ret) break;
    Where(fd, "  caller", ret);
    if (next <= fp) break;
    fp = next;
  }
  close(fd);
#endif
}
struct Probe {
  Probe() {
    struct sigaction a{};
    a.sa_sigaction = ProbeHandler;
    a.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_RESTART;
    sigemptyset(&a.sa_mask);
    sigaction(SIGUSR2, &a, nullptr);
    std::thread([] {
      std::vector<std::pair<int, unsigned long>> last;
      for (;;) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        if (access("stackprobe", F_OK) != 0) { last.clear(); continue; }
        std::vector<std::pair<int, unsigned long>> now;
        if (DIR* d = opendir("/proc/self/task")) {
          while (dirent* e = readdir(d)) {
            const int tid = atoi(e->d_name);
            if (tid <= 0) continue;
            char p[64], buf[512] = {};
            snprintf(p, sizeof p, "/proc/self/task/%d/stat", tid);
            const int fd = open(p, O_RDONLY);
            if (fd < 0) continue;
            (void)!read(fd, buf, sizeof buf - 1);
            close(fd);
            const char* r = strrchr(buf, ')');
            if (!r) continue;
            unsigned long ut = 0, st = 0;
            // fields after ')': state(3) ... utime(14) stime(15)
            int field = 2;
            const char* q = r + 1;
            while (*q && field < 14) { if (*q == ' ') ++field; ++q; }
            sscanf(q, "%lu %lu", &ut, &st);
            now.emplace_back(tid, ut + st);
          }
          closedir(d);
        }
        for (auto& [tid, t] : now)
          for (auto& [ltid, lt] : last)
            if (tid == ltid && t - lt >= 5 && tid != int(gettid())) syscall(SYS_tgkill, getpid(), tid, SIGUSR2);
        last = now;
      }
    }).detach();
  }
} g_probe;

struct Install {
  Install() {
    for (int sig : {SIGABRT, SIGSEGV, SIGBUS, SIGILL, SIGFPE}) {
      struct sigaction a{};
      a.sa_sigaction = Handler;
      a.sa_flags = SA_SIGINFO | SA_ONSTACK;
      sigemptyset(&a.sa_mask);
      g_old_valid[sig] = sigaction(sig, &a, &g_old[sig]) == 0;
    }
  }
} g_install;
}  // namespace
