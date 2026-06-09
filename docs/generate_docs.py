import csv
import os
import sys
from collections import defaultdict

def parse_csv_to_rst_tables(csv_file):
    """Parse CSV file and create RST tables grouped by module (first word before .)"""
    modules = defaultdict(dict)
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            for row in reader:
                variable = row.get('Variable', '').strip()
                units = row.get('Units', '').strip()
                speedy = row.get('Speedy', '').strip()
                description = row.get('Description', '').strip()
                
                if not variable:
                    continue
                
                if '.' in variable:
                    parts = variable.split('.')
                    module = parts[0]
                    sub_var = parts[1]
                else:
                    module = 'PhysicsState'
                    sub_var = variable
                
                if sub_var not in modules[module]:
                    modules[module][sub_var] = {
                        'units': units,
                        'speedy': speedy,
                        'description': description
                    }
    
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)
    
    # Generate RST content
    rst_content = generate_sphinx_rst(modules)
    return rst_content

def generate_sphinx_rst(modules):
    rst_content = """Speedy Variable Translation
===========================

"""
    for module_name in sorted(modules.keys()):
        rst_content += create_sphinx_rst_table(module_name, modules[module_name])
        rst_content += "\n"
    
    return rst_content

def create_sphinx_rst_table(module_name, data):
    if not data:
        return f"{module_name}\n{'-' * len(module_name)}\n\nNo variables available for this module.\n\n"
    
    module_display_name = module_name.capitalize()
    rst_table = f"{module_display_name}\n"
    rst_table += "-" * len(module_display_name) + "\n\n"
    
    rows = []
    for sub_var in sorted(data.keys()):
        row_data = data[sub_var]
        rows.append([
            sub_var,
            row_data['units'] if row_data['units'] else 'N/A',
            row_data['speedy'] if row_data['speedy'] else 'N/A', 
            row_data['description'] if row_data['description'] else 'No description available'
        ])
    
    rst_table += ".. list-table::\n"
    rst_table += "   :header-rows: 1\n"
    rst_table += "   :widths: 25 15 20 40\n\n"
    rst_table += "   * - Jax Variable\n"
    rst_table += "     - Units\n"
    rst_table += "     - Speedy Equivalent\n"
    rst_table += "     - Description\n"
    
    for row in rows:
        rst_table += f"   * - ``{row[0]}``\n"
        rst_table += f"     - {row[1]}\n"
        rst_table += f"     - ``{row[2]}``\n"
        rst_table += f"     - {row[3]}\n"
    
    rst_table += "\n"
    return rst_table

def update_sphinx_doc(csv_file, output_file):
    """Update the Sphinx documentation file"""
    # Generate RST content
    rst_content = parse_csv_to_rst_tables(csv_file)
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(rst_content)
        return True
    except Exception:
        return False

# Repo-relative default paths computed from this file's location, so the
# generator behaves identically whether it is run by hand, by the Sphinx build
# (docs/source/conf.py) on Read the Docs, or in CI — regardless of the current
# working directory.
_DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_DOCS_DIR)
DEFAULT_CSV_PATH = os.path.join(
    _REPO_ROOT, 'jcm', 'physics', 'speedy', 'units_table.csv')
DEFAULT_OUTPUT_FILE = os.path.join(_DOCS_DIR, 'source', 'speedy_translation.rst')


def generate(csv_path=None, output_file=None):
    """Generate the SPEEDY variable-translation page from the units CSV.

    Writes ``output_file`` and returns the number of variables documented.
    This is the single entry point used both by the CLI (:func:`main`) and by
    the Sphinx build (``docs/source/conf.py``), so the page is produced at
    build time and never committed to the repository (see issue #394).
    """
    csv_path = csv_path or DEFAULT_CSV_PATH
    output_file = output_file or DEFAULT_OUTPUT_FILE
    if not update_sphinx_doc(csv_path, output_file):
        raise RuntimeError(f"Failed to generate {output_file} from {csv_path}")
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return sum(1 for row in reader if row.get('Variable', '').strip())


def main():
    """Execute workflow to update Sphinx documentation from CSV."""
    total_vars = generate()
    print(f"Updated documentation for {total_vars} variables.")


if __name__ == "__main__":
    main()