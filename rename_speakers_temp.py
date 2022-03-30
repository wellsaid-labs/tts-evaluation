import pathlib

import pandas


def fix_speaker_def():
    file_ = pathlib.Path("run/data/_loader/english/wsl.py")
    text = file_.read_text()
    csv = pathlib.Path("/Users/michaelp/Downloads/Stock Voices Master List - English.csv")
    data_frame = pandas.read_csv(csv)
    for _, row in data_frame.iterrows():
        uppercase_va = str(row["VA"]).upper().replace(" ", "_").replace(".", "")
        uppercase_avatar = str(row["Avatar"]).upper().replace(" ", "_").replace(".", "")
        text = text.replace(uppercase_va, uppercase_avatar)

        text = text.replace(str(row["VA"]), str(row["Avatar"]).replace(".", ""))

    for line in text.split("\n"):
        if len(line) > 0 and line[-1] == ")" and "make(" in line:
            var_name, call = tuple(line.split(" = "))
            print(call)
            open, close = tuple(call.split('", "'))
            gcs_dir = f'{open.split("(")[-1]}"'
            name = f'"{close[:-1]}'
            new_line = f'{var_name} = make("{var_name.lower()}", {name}, {gcs_dir})'
            text = text.replace(line, new_line)

    for line in text.split("\n"):
        if len(line) > 0 and line[-1] == ")" and "make(" in line:
            var_name, call = tuple(line.split(" = "))
            print(call)
            open, close = tuple(call.split('", "'))
            gcs_dir = f'{open.split("(")[-1]}"'
            name = f'"{close[:-1]}'
            new_line = f'{var_name} = make("{var_name.lower()}", {name}, {gcs_dir})'
            text = text.replace(line, new_line)

    file_ = pathlib.Path("run/data/_loader/english/wsl_new.py").write_text(text)


def fix_web_links():
    file_ = pathlib.Path("run/deploy/worker.py")
    text = file_.read_text()
    csv = pathlib.Path("/Users/michaelp/Downloads/Stock Voices Master List - English.csv")
    data_frame = pandas.read_csv(csv)
    for _, row in data_frame.iterrows():
        uppercase_va = str(row["VA"]).upper().replace(" ", "_").replace(".", "")
        uppercase_avatar = str(row["Avatar"]).upper().replace(" ", "_").replace(".", "")
        text = text.replace(uppercase_va, uppercase_avatar)

        text = text.replace(str(row["VA"]), str(row["Avatar"]).replace(".", ""))

    file_ = pathlib.Path("run/deploy/worker_new.py").write_text(text)


fix_web_links()
