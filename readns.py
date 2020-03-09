import analysis_scripts as asc
import neuroshare as ns

fpath = '/home/yc/share/1_fff_gauss1blink.mcd'

f = ns.File(fpath)
threshold = 75
monitor_delay = 0.025 # in seconds
if f.entity_count == 256:
    meatype = 252
elif f.entity_count == 63:
    meatype = 60
else:
    ValueError(f'Unknown MEA type with {f.entity_count}')
#sampling_rate =int(1/f.time_stamp_resolution)

voltage, time, length = f.get_entity(meatype).get_data() # units: V, s, no unit

voltage = voltage*1000 # Convert to milivolts
voltage -= voltage[voltage<75].mean() # set baseline to zero

time += monitor_delay   # There seems to be a mismatch between the units in the
                    		# previous code HINT

onsets, offsets = asc.detect_threshold_crossing(voltage, threshold)

ft_on = time[onsets]
ft_off = time[offsets]

