#!bin/zsh

__WSL_ENVS_PATH="disk/envs"
__WSL_OLD_ENV=""
__WSL_PREV_ENV=""
__WSL_VENV=""

function _wsl_precmd() {
    # Find changes to environment since last command and save them to disk.
    # NOTE: Avoid saving changes to special environment variable `PS1`.
    comm -13 <(echo "$__WSL_PREV_ENV" ) <(echo "$(export -p)") | grep -v ' PS1=' >> $__WSL_VENV
    # Apply current variables saved in disk.
    . $__WSL_VENV
    __WSL_PREV_ENV=$(export -p)
}

function _wsl_preexec() {
    . $__WSL_VENV
    __WSL_PREV_ENV=$(export -p)
}


function vars() {
    case $1 in

      help)
        echo "`vars` sync variables accross multiple consoles via a virtual environment."
        echo ""
        echo "Usage: vars COMMAND [ARGS]..."
        echo ""
        echo "Commands:"
        echo "  make        Create an ENVIRONMENT."
        echo "  activate    Activate ENVIRONMENT and sync variables."
        echo "  deactivate  Deactivate current environment and reset variables."
        echo "  delete      Delete ENVIRONMENT."
        echo ""
        echo "Example:"
        echo "$ . run/utils/vars.sh"
        echo "$ vars make vars-env"
        echo "$ vars activate vars-env"
        echo "$ export VARIABLE=value"
        echo "$ vars deactivate"
        echo "$ vars delete vars-env"
        ;;

      make)
        ENV_PATH="$__WSL_ENVS_PATH/$2"
        if [ -n "$2" ]; then
            if [ -f "$ENV_PATH" ]; then
                echo "Environment '$2' already exists."
            else
                :>> "$__WSL_ENVS_PATH/$2"
            fi
        else
            echo "Please pass in a ENVIRONMENT name."
        fi
        ;;

      activate)
        ENV_PATH="$__WSL_ENVS_PATH/$2"
        if [ -f "$ENV_PATH" ]; then
            __WSL_OLD_ENV=$(export -p)
            __WSL_PREV_ENV=$(export -p)
            __WSL_VENV=$ENV_PATH
            # Set bash terminal to indicate the virtual environment.2
            export PS1="($2) ${PS1:-}"
            _wsl_precmd
            _wsl_preexec
            precmd_functions+=(_wsl_precmd)
            preexec_functions+=(_wsl_preexec)
        else
            echo "Environment $2 does not exist."
        fi
        ;;

      deactivate)
        if [[ ! -v __WSL_VENV ]]; then
            echo "Environment is not active."
        else
            # Unset any variables set in `__WSL_VENV`
            unset $(awk -F '[ =]' '{print $2}' $__WSL_VENV | tr '\n' ' ')
            # Reset any variables to original values
            eval "$__WSL_OLD_ENV";
            # Remove `_wsl_precmd` and `_wsl_preexec` functions from function list.
            precmd_functions=(${precmd_functions:#_wsl_precmd})
            preexec_functions=(${preexec_functions:#_wsl_preexec})
            __WSL_OLD_ENV=""
            __WSL_PREV_ENV=""
            __WSL_VENV=""
        fi
        ;;

      delete)
        ENV_PATH="$__WSL_ENVS_PATH/$2"
        if [ -n "$2" ]; then
            if [ -f "$ENV_PATH" ]; then
                rm $ENV_PATH
            else
                echo "Environment $2 does not exist."
            fi
        else
            echo "Please pass in a ENVIRONMENT name."
        fi
        ;;

      *)
        echo -n "Command not found."
        ;;
    esac
}


