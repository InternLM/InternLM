import os

# obtain the directory
current_script_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_script_path)
# read template content
template_file_path = current_directory +  "/30b.template"
with open(template_file_path, "r") as template_file:
    template_content = template_file.read()

# define variable
sequence_parallel = ["True", "False"]
num_chunks = [1, 2]
num_gpus = [8, 16]

parallel_configs = dict()

for gpu in num_gpus:
    parallel_configs_sub = []
    if gpu == 8:
        step = 2
    else:
        step = 4
    for it in range(0, gpu + 1, step):
        for j in range(0, gpu + 1, step):
            if it == 0 and j == 0:
                parallel_configs_sub.append((1, 1, gpu))
            elif it == 0:
                if gpu % j != 0:
                    continue
                parallel_configs_sub.append((1, j, gpu // j))
            elif j == 0:
                if gpu % it != 0:
                    continue
                parallel_configs_sub.append((it, 1, gpu // it))
            else:
                if it * j > gpu or gpu % (it * j) != 0:
                    continue
                parallel_configs_sub.append((it, j, gpu // it // j))
    parallel_configs[gpu] = (parallel_configs_sub)

table_content = "id\tgpu\tsp\tchunk\ttp\tpp\tdp\n"

new_directory = current_directory + "/30b_configs"
os.makedirs(new_directory, exist_ok=True)

# generate all configs
idx = 0
all_config_contents = []
for sequence_parallel in sequence_parallel:
    for num_chunk in num_chunks:
        for key, parallel in parallel_configs.items():
            for tensor_size, pipeline_size, dp in parallel:
                replacement_values = {
                    "JOB_NAME": str(f"30B_Train_{idx}"),
                    "SEQUENCE_PARALLEL": sequence_parallel,
                    "NUM_CHUNKS": str(num_chunk),
                    "TENSOR_SIZE": str(tensor_size),
                    "PIPELINE_SIZE": str(pipeline_size)
                }

                # update template content
                config_content = template_content
                for variable, value in replacement_values.items():
                    config_content = config_content.replace(f"${{{variable}}}", value)
                
                all_config_contents.append(config_content)
                idx = idx + 1
                
                # update config table
                table_content += f"{idx}\t{key}\t{sequence_parallel}\t{num_chunk}\t{tensor_size}\t{pipeline_size}\t{dp}\n"

# write all configs into config file
for idx, config_content in enumerate(all_config_contents):
    output_file_path = current_directory + f"/30b_configs/30b_config_{idx}.py"
    with open(output_file_path, "w") as output_file:
        output_file.write(config_content)

# write the table
output_table_path = current_directory + "/30b_configs/config_table.txt"
with open(output_table_path, "w") as output_table:
    output_table.write(table_content)

print("Generate configs done!")
