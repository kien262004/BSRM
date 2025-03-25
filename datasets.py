filename = './instance/testcase.txt'

with open(filename, 'r') as f:
    lines = f.readlines()
    output_path = './instance/case'
    file_output = None
    i = 1
    for line in lines:
        if line.startswith('Case'):
            file_output = output_path + f'{i}.txt'
            i += 1
        else:
            with open(file_output, 'a+') as fout:
                fout.write(line)