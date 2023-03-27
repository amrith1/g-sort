import xml.etree.ElementTree as ET

class VisionConfigXML:

    def __init__ (self,
                  array_part : int,
                  array_n_parts : int,
                  flip_x : bool,
                  flip_y: bool) -> None:

        self.array_part = array_part
        self.array_n_parts = array_n_parts
        self.flip_x = flip_x
        self.flip_y = flip_y

    @classmethod
    def parse_from_xml_path (cls,
                             xml_path : str) -> 'VisionConfigXML':

        array_part, array_n_part, flip_x, flip_y = None, None, None, None

        tree = ET.parse(xml_path)
        root = tree.getroot()
        for child in root:
            if child.attrib['name'] == 'Spike Finding':
                for group in child:
                    if group.attrib['name'] == 'Set Electrodes':
                        for param in group:

                            if param.attrib['name'] == 'flipX':
                                flip_x = (param.attrib['value'] == 'true')
                            elif param.attrib['name'] == 'flipY':
                                flip_y = (param.attrib['value'] == 'true')
                            elif param.attrib['name'] == 'arrayPart':
                                array_part = int(param.attrib['value'])
                            elif param.attrib['name'] == 'arrayNParts':
                                array_n_part = int(param.attrib['value'])

        return VisionConfigXML(array_part,
                               array_n_part,
                               flip_x,
                               flip_y)


    def __str__ (self):

        return 'VisionConfigXML({0},{1},{2},{3})'.format(self.array_part,
                                                          self.array_n_parts,
                                                          self.flip_x,
                                                          self.flip_y)


if __name__ == '__main__':

    print(VisionConfigXML.parse_from_xml_path('/Users/wueric/workremote/primate.xml'))
