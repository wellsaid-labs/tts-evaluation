# Gunicorn configuration file
#
# WARNING: this file is moved to the default gunicorn config file location during our docker build
# phase.
#
# NOTE: CLI arguments may override configurations defined in this file. See the following link for
# argument precedence: https://docs.gunicorn.org/en/stable/settings.html.

# @see https://docs.gunicorn.org/en/stable/settings.html#access-log-format
access_log_format = '%(h)s %(l)s %(t)s "%(r)s" %(s)s text_length[%({x-text-length}o)s] '
access_log_format += 'response_time_ms[%(M)s] response_size_bytes[%(b)s] "%(a)s"'

# TODO: leaving this here as a potential consideration for speeding up the model initialization
# in each worker.
#
# def post_fork(server, worker):
#     pass
# def post_worker_init(worker):
#     pass
