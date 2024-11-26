import os
import shutil
import re
def add_spaces_to_camel_case(text):
    # Use regular expression to find uppercase letters and add space before them
    spaced_text = re.sub(r'(?<!^)([A-Z])', r' \1', text)
    return spaced_text

# Python script to generate HTML file with a dropdown for selecting HTML files
filenames_dct ={"State":{},
                "State & local":{},
                "Local":{},
                "EFNA":{},
                "PersonalIncome":{},
                "StateRetirementPlans":{},
                "NDOilProduction":{},
                "Employment":{},
                "PropertyTaxesAndHomePrices":{},
                }
root = "../outputs"
for state_local in filenames_dct.keys():
    filenames_dct[state_local] = {}
    for path, directory, filenames in os.walk(root):
        filenames = sorted(filenames)
        # filenames = [f for f in filenames if "area" in f.lower() and "percent" in f.lower() and ("percent of general revenue" in f.lower() or "percent of expenditure" in f.lower()) and ("expenditures" in f.lower() or "taxes" in f.lower())] 
        other_sl = [sl == path[len(root)+1:len(root) + 1 + len(sl)] for sl in filenames_dct.keys() if len(sl) > len(state_local)] 
        if state_local == path[len(root)+1:len(root) + 1 + len(state_local)] and True not in other_sl:
            print(path)
            filenames_dct[state_local] = filenames
            shutil.copytree(path, f"InteractiveHTML", dirs_exist_ok=True)
        
folder = "InteractiveHTML"
# filenames_dct = {key: [f"{folder}/{f}" for f in filenames] for key, filenames in filenames_dct.items()}
# filenames_dct["Taxonomies"] = [f"{f} Taxonomy.html" for f in filenames_dct["Taxonomies"]]
# List of filenames to include in the dropdown
for key, filenames in filenames_dct.items():
    if key not in ["EFNA", "PersonalIncome", "Employment"]:
        key = key + "Finances"
    title = add_spaces_to_camel_case(key).strip().title()
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                padding: 20px;
            }}
            iframe {{
                width: 100%;
                height: 1000px;
                border: 1px solid #ccc;
                margin-top: 20px;
            }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        
        <label for="pageSelect">Type: </label>
        <select id="pageSelect">
            <option value="" disabled selected>Select a page</option>
    """

    # Dynamically generate dropdown options from the filenames list
    for filename in filenames:
        html_content += f'        <option value="{folder}/{filename}">{filename.replace(".html","")}</option>\n'

    html_content += """
        </select>

        <iframe id="contentFrame" src=""></iframe>

        <script>
            document.getElementById('pageSelect').addEventListener('change', function() {
                var selectedPage = this.value;
                document.getElementById('contentFrame').src = selectedPage;
            });
        </script>

    </body>
    </html>
    """
    fname = f"index{key}.html"
    # Write the HTML content to a file
    with open(fname, "w") as file:
        file.write(html_content)

    print(f"HTML file '{fname}' has been generated successfully.")