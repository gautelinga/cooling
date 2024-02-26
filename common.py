from mpi4py import MPI
mpi_rank = MPI.COMM_WORLD.Get_rank()

RED = "\033[1;37;31m{s}\033[0m"
BLUE = "\033[1;37;34m{s}\033[0m"
GREEN = "\033[1;37;32m{s}\033[0m"
YELLOW = "\033[1;37;33m{s}\033[0m"
CYAN = "\033[1;37;36m{s}\033[0m"
NORMAL = "{s}"
ON_RED = "\033[41m{s}\033[0m"

def info_style(message, check=True, style=NORMAL):
    if mpi_rank == 0 and check:
        print(style.format(s=message))


def info_red(message, check=True):
    info_style(message, check, RED)


def info_blue(message, check=True):
    info_style(message, check, BLUE)


def info_yellow(message, check=True):
    info_style(message, check, YELLOW)


def info_green(message, check=True):
    info_style(message, check, GREEN)


def info_cyan(message, check=True):
    info_style(message, check, CYAN)


def info(message, check=True):
    info_style(message, check)


def info_on_red(message, check=True):
    info_style(message, check, ON_RED)


def info_split_style(msg_1, msg_2, style_1=BLUE, style_2=NORMAL, check=True):
    if mpi_rank == 0 and check:
        print(style_1.format(s=msg_1) + " " + style_2.format(s=msg_2))


def info_split(msg_1, msg_2, check=True):
    info_split_style(msg_1, msg_2, check=check)


def info_warning(message, check=True):
    info_split_style("Warning:", message, style_1=ON_RED, check=check)


def info_error(message, check=True):
    info_split_style("Error:", message, style_1=ON_RED, check=check)
    exit("")