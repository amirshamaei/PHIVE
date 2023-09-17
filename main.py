import json
import engine as eng

def main():
    file = open('training_report.csv', 'w+')
    json_file_path = 'runs/exp46.json'
    with open(json_file_path, 'r') as j:
        contents = json.loads(j.read())
    for run in contents['runs']:
        if run["version"] in ["1/"]:
            engine = eng . Engine(run)
            engine . dotrain(run["ens_id"])
            # engine.dotest()
    file.close()

if __name__ == '__main__':
    main()