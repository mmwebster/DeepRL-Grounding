import wandb
import argparse

# @brief return hours float from a string formatted "Time HHh MMm SSs"
def parse_hrs_float(time_str):
    time_str = time_str.split('Time ')[1]
    time_hrs = int(time_str.split('h ')[0])
    time_str_rem = time_str.split('h ')[1]
    time_mins = int(time_str_rem.split('m ')[0])
    time_str_rem = time_str_rem.split('m ')[1]
    time_secs = int(time_str_rem.split('s')[0])
    hrs_float = time_hrs + (time_mins/60) + (time_secs/60/60)
    return hrs_float

# @brief return average accuracy float from a string formatted "Avg Accuracy X.XX"
def parse_avg_acc_float(avg_acc_str):
    avg_acc_str = avg_acc_str.split('Avg Accuracy ')[1]
    return float(avg_acc_str)

def is_train_status_line(line):
    return "Time" in line

def parse_avg_reward_float(s):
    return float(s.split('Avg Reward ')[1])

def parse_avg_ep_len(s):
    return float(s.split('Avg Ep length ')[1])

def parse_best_reward(s):
    return float(s.split('Best Reward ')[1])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Plot from logs")
    parser.add_argument('-f', '--filename', type=str, required=True)
    args = parser.parse_args()

    wandb.init(project="rl-language-grounding")

    with open(args.filename, 'r') as f:
        log_lines = f.readlines()
        for line in log_lines:

            #fields = [field_raw.split()]
            #print(line)

            if is_train_status_line(line):
                # extract fields
                line = line.split('\n')[0]
                fields_raw = line.split(',')
                hrs = parse_hrs_float(fields_raw[0])
                avg_reward = parse_avg_reward_float(fields_raw[1])
                avg_acc = parse_avg_acc_float(fields_raw[2])
                avg_ep_len = parse_avg_ep_len(fields_raw[3])
                best_reward = parse_best_reward(fields_raw[4])
                # log fields to wandb
                d = {'hrs': hrs, 'avg_accuracy': avg_acc, 'avg_reward': avg_reward, 'avg_ep_len': avg_ep_len, 'best_reward': best_reward}
                wandb.log(d)

                #print(f"Data: {d}")
