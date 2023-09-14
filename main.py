import csv
import json
from matplotlib import pyplot as pl
import engine as eng

def main():
    file = open('training_report.csv', 'w+')
    writer = csv.writer(file)
    json_file_path = 'runs/exp47.json'
    with open(json_file_path, 'r') as j:
        contents = json.loads(j.read())
    for run in contents['runs']:
        if run["version"] in ["1/"]:
        #     print(run["versiosn"])
            # try:
        #     print(":)")
        # else:
        #     run["ens"] = 1
        # run["numOfSample"] = 1000
            engine = eng . Engine(run)
                # engine.tuner()
            engine . dotrain(0)
          # writer.writerows([run["child_root"],run["version"], "%100"])
        #     plt.close('all')
        #     engine . simulation()
        #     engine.simulation_mc()
        #     engine.dotest()
        # except:
        # writer.writerows([run["child_root"],run["version"], "Failed"])

    file.close()
if __name__ == '__main__':
    main()