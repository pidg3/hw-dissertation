names = """word_freq_make:         continuous.
word_freq_address:      continuous.
word_freq_all:          continuous.
word_freq_3d:           continuous.
word_freq_our:          continuous.
word_freq_over:         continuous.
word_freq_remove:       continuous.
word_freq_internet:     continuous.
word_freq_order:        continuous.
word_freq_mail:         continuous.
word_freq_receive:      continuous.
word_freq_will:         continuous.
word_freq_people:       continuous.
word_freq_report:       continuous.
word_freq_addresses:    continuous.
word_freq_free:         continuous.
word_freq_business:     continuous.
word_freq_email:        continuous.
word_freq_you:          continuous.
word_freq_credit:       continuous.
word_freq_your:         continuous.
word_freq_font:         continuous.
word_freq_000:          continuous.
word_freq_money:        continuous.
word_freq_hp:           continuous.
word_freq_hpl:          continuous.
word_freq_george:       continuous.
word_freq_650:          continuous.
word_freq_lab:          continuous.
word_freq_labs:         continuous.
word_freq_telnet:       continuous.
word_freq_857:          continuous.
word_freq_data:         continuous.
word_freq_415:          continuous.
word_freq_85:           continuous.
word_freq_technology:   continuous.
word_freq_1999:         continuous.
word_freq_parts:        continuous.
word_freq_pm:           continuous.
word_freq_direct:       continuous.
word_freq_cs:           continuous.
word_freq_meeting:      continuous.
word_freq_original:     continuous.
word_freq_project:      continuous.
word_freq_re:           continuous.
word_freq_edu:          continuous.
word_freq_table:        continuous.
word_freq_conference:   continuous.
char_freq_semicolon:            continuous.
char_freq_open_bracket:            continuous.
char_freq_square_bracket:            continuous.
                      char_freq_exclamation:            continuous.
                      char_freq_dollar:            continuous.
                      char_freq_hash:            continuous.
                      capital_run_length_average: continuous.
                      capital_run_length_longest: continuous.
                      capital_run_length_total:   continuous."""


def _generate_single_item(name):
    if name == 'class':
      return {
          'name': 'class',
          'type': 'outcome'
      }
    elif 'freq' in name:
      return {
          'name': name,
          'type': 'numerical',
          'used': True,
          'min': 0,
          'max': 100
      }
    else:
      return {
          'name': name,
          'type': 'numerical',
          'used': True,
          'min': 0,
      }

def get_names(): 
  split_names = names.split()
  names_only = list(filter(lambda text: text != 'continuous.', split_names))
  colon_removed = list(map(lambda text: text[:-1], names_only))
  colon_removed.append('class')
  return colon_removed

def get_metadata():

  names = get_names()
  return list(map(_generate_single_item, names))

