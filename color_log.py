import logging


# ANSI escape sequences for colors
class AnsiColorCode:
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    RESET = '\033[0m'


# Custom logger to add color based on log level
class ColorLogger(logging.Logger):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def warning(self, msg, *args, **kwargs):
        print(f"{AnsiColorCode.YELLOW}WARNING: {msg} {str(args).strip('()')}{AnsiColorCode.RESET}")

    def info(self, msg, *args, **kwargs):
        print(f"{AnsiColorCode.GREEN}INFO: {msg} {str(args).strip('()')}{AnsiColorCode.RESET}")

if __name__ == '__main__':
    # Create a custom logger instance
    logger = ColorLogger()

    # Example usage
    logger.warning("sllang is not installed. If you are not using sglan, you can ignore this warning.")
    logger.info("get_predictor cost: 1.55s")
