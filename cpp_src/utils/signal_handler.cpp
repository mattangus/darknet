#include <stdio.h>
#include <signal.h>
#include <stdio.h>
#include <signal.h>
#include <execinfo.h>
#include <cstdlib>
#include <sstream>
#include <iostream>
#include <regex>

#include "signal_handler.hpp"

static const int all_signals[] = {
    #ifdef SIGHUP
        SIGHUP,  /* POSIX.1 */
    #endif
    #ifdef SIGQUIT
        SIGQUIT, /* POSIX.1 */
    #endif
    #ifdef SIGTRAP
        SIGTRAP, /* POSIX.1 */
    #endif
    #ifdef SIGIO
        SIGIO,   /* BSD/Linux */
    #endif
    #ifdef SIGKILL
        SIGKILL,
    #endif
    #ifdef SIGSYS
        SIGSYS,
    #endif
    #ifdef SIGPIPE
        SIGPIPE,
    #endif
    #ifdef SIGALRM
        SIGALRM,
    #endif
    #ifdef SIGSTOP
        SIGSTOP,
    #endif

    /* C89/C99/C11 standard signals: */
    
    #ifdef SIGABRT
        SIGABRT,
    #endif
    #ifdef SIGFPE
        SIGFPE,
    #endif
    #ifdef SIGINT
        SIGINT,
    #endif
    #ifdef SIGILL
        SIGILL,
    #endif
    SIGSEGV,
    SIGTERM
};

std::string exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

typedef struct sigcontext sigcontext;

void bt_sighandler(int sig, sigcontext ctx) {

    void *trace[30];
    char **messages = (char **)NULL;
    int i, trace_size = 0;

    if (sig == SIGSEGV)
        std::cerr << "Got signal " << sys_siglist[sig] << ", faulty address is " << (void*)ctx.cr2 << " from " << (void*)ctx.rip << std::endl;
    else
        std::cerr << "Got signal " << sys_siglist[sig] << std::endl;

    trace_size = backtrace(trace, 30);
    /* overwrite sigaction with caller's address */
    trace[1] = (void *)ctx.rip;
    messages = backtrace_symbols(trace, trace_size);

    std::string restr = "([\\w\\/\\.]+)\\(\\+(0[xX][0-9a-fA-F]+)\\)";
    std::regex re(restr);

    /* skip first stack frame (points here) */
    for (i=1; i<trace_size; ++i)
    {
        std::cout << "[bt] #" << i << " " << messages[i] << std::endl;

        std::string current(messages[i]);

        std::smatch m;
        if (std::regex_search(current, m, re))
        {
            std::stringstream ss;
            ss << "addr2line " << m[2].str() << " -e " << m[1].str();
            std::string cmd = ss.str();
            std::cout << "       ";
            std::string istr = std::to_string(i);
            for(int j = 0; j < istr.size(); j++)
                std::cout << " ";
            std::cout << exec(cmd.c_str()); // end line from addr2line
        }

    }

    exit(0);
}


void register_handlers()
{
    struct sigaction act;

    act.sa_handler = (__sighandler_t)bt_sighandler;
    sigemptyset(&act.sa_mask);
    act.sa_flags = SA_RESTART;

    int num_sig = sizeof(all_signals) / sizeof(all_signals[0]);

    for(int i = 0; i < num_sig; i++)
    {
        if (sigaction(all_signals[i], &act, NULL)) {
            fprintf(stderr, "Cannot install signal %d.\n", all_signals[i]);
        }
    }
}