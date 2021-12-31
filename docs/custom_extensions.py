from pybtex.style.formatting.unsrt import Style as UnsrtStyle
from pybtex.style.labels.alpha import LabelStyle as AlphaLabelStyle


class AuthorYearLabelStyle(AlphaLabelStyle):
    def format_label(self, entry):
        return entry.persons["author"][0].last_names[0] + entry.fields["year"]
        # return str(entry)


class AuthorYearStyle(UnsrtStyle):
    default_label_style = "authoryear"
