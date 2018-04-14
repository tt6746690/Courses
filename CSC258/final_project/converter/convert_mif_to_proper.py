"""
e.g. python3.6 convert_mif_to_proper.py mif/background.mono.mif mif/converted_background_mono.mono.mif
"""

import sys

def get_text(to_convert):
    with open(to_convert, "r") as f:
        text = f.readlines()
    return text

def collect_results(text):
    results = {}
    for i in range(len(text)):
        if 5 < i < len(text) - 1:
            temp_list = text[i].split(" ")
            # clean key and end
            temp_list[0] = temp_list[0].strip("\t")
            temp_list[-1] = temp_list[-1].strip(";\n")
            results[temp_list[0]] = temp_list[2:]

    return results

def build_final_list(results):
    final_result = {}

    for i in results:
        try:
            curr = int(i)
            for j in results[str(i)]:
                final_result[curr] = int(j)
                curr += 1
        except:
            pass
    return final_result

def build_final_text(final, output_name):
    final_text = []
    final_text += text[:6]

    for i in final_result:
        final_text.append(str(i) + " : " + str(final_result[i]) + ";\n")

    with open(output_name, "w") as f:
        for i in final_text:
            f.write(i)
        f.write("End;\n")

if __name__ == "__main__":

    input_mif = sys.argv[1]
    output_name = sys.argv[2]

    text = get_text(input_mif)
    results = collect_results(text)
    final_result = build_final_list(results)
    build_final_text(final_result, output_name)
