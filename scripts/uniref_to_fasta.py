import argparse
import xml.etree.ElementTree as ET

from tqdm import tqdm


def convert_uniref_to_fasta(xml_file, fasta_file):
    """
    Converts a UniRef XML file to FASTA format.

    This function parses the XML file iteratively, which is memory-efficient
    for large files. It extracts the representative member's sequence from
    each entry and writes it to a FASTA file. A progress bar is displayed.

    Args:
        xml_file (str): Path to the input UniRef XML file.
        fasta_file (str): Path to the output FASTA file.
    """
    # The namespace is defined in the XSD file.
    # All tags are prefixed with this namespace.
    namespace = "{http://uniprot.org/uniref}"
    entry_tag = f"{namespace}entry"
    name_tag = f"{namespace}name"
    rep_member_tag = f"{namespace}representativeMember"
    # db_ref_tag = f"{namespace}dbReference"
    sequence_tag = f"{namespace}sequence"

    # Use iterparse for memory-efficient streaming of the large XML file.
    # We are interested in the 'end' event for each 'entry' element.
    context = ET.iterparse(xml_file, events=("end",))

    with open(fasta_file, "w") as f_out, tqdm(desc="Processing entries") as pbar:
        for event, elem in context:
            # When we have finished parsing an 'entry' element...
            if elem.tag == entry_tag:
                try:
                    # Extract the entry ID and name.
                    entry_id = elem.attrib["id"]
                    entry_name = elem.find(name_tag).text

                    # Find the representative member.
                    rep_member = elem.find(rep_member_tag)

                    # Find the sequence within the representative member.
                    sequence_elem = rep_member.find(sequence_tag)

                    if sequence_elem is not None and sequence_elem.text:
                        sequence = sequence_elem.text

                        # Construct the FASTA header.
                        header = f">{entry_id} {entry_name}"

                        # Write the FASTA entry to the output file.
                        f_out.write(header + "\n")
                        f_out.write(sequence + "\n")

                except (AttributeError, KeyError) as e:
                    # Handle cases where an expected element or attribute is missing.
                    print(f"Skipping malformed entry: {ET.tostring(elem, 'utf-8')}. Error: {e}")

                finally:
                    # Clear the element from memory to free up resources.
                    # This is crucial for processing large files.
                    elem.clear()
                    # Update the progress bar.
                    pbar.update(1)


if __name__ == "__main__":
    # Set up command-line argument parsing.
    parser = argparse.ArgumentParser(description="Convert UniRef XML file to FASTA format.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("xml_file", help="Path to the input UniRef XML file.")
    parser.add_argument("fasta_file", help="Path to the output FASTA file.")

    args = parser.parse_args()

    print(f"Converting {args.xml_file} to {args.fasta_file}...")
    convert_uniref_to_fasta(args.xml_file, args.fasta_file)
    print("Conversion complete.")
