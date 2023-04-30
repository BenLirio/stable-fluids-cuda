import random
import sfc
import timings
import plotly.figure_factory as ff

GANTT_CHART_SCALE = 1

if __name__ == '__main__':
  config = sfc.get_config(output=sfc.OUTPUT_PERFORMANCE)
  logs = timings.generate_timings(config)
  logs_with_depth = [ x for x in logs if 'DEPTH' in x ]
  log_pairs = timings.as_log_pairs(logs_with_depth)


  flame_graph_data = []
  def get_resource(log):
    if 'SOURCE' in log or 'SINK' in log: return 'SOURCE/SINK'
    if 'DIFFUSE' in log: return 'DIFFUSE'
    if 'ADVECT' in log: return 'ADVECT'
    if 'PROJECT' in log: return 'PROJECT'
    if 'COLOR' in log: return 'COLOR'
    if 'VELOCITY' in log: return 'VELOCITY'
    if 'STEP' in log: return 'STEP'

  for [log_start, log_end] in log_pairs:
    flame_graph_data.append({
      'Resource': get_resource(log_start),
      'Start': log_start['TIME']*GANTT_CHART_SCALE,
      'Finish': log_end['TIME']*GANTT_CHART_SCALE,
      'Task': log_start['DEPTH'],
    })

  fig = ff.create_gantt(
    flame_graph_data,
    index_col="Resource",
    show_colorbar=True,
    group_tasks=True,
  )

  times = [ x['TIME'] for x in logs_with_depth ]
  start_time = min(times)
  end_time = max(times)

  fig.update_layout(xaxis=dict(
    title='Time (seconds)',
    range=[start_time*GANTT_CHART_SCALE*0.9, end_time*GANTT_CHART_SCALE*1.01],
    constrain='domain',
    tickformat='%S.%L'
  ))
  fig.update_layout(yaxis=dict(
    visible=False,
    showticklabels=False,
    showgrid=False,
    zeroline=False,
  ))
  fig.update_layout(title_text=f'Flame Chart')

  # adds subtitle
  fig.add_annotation(
    xref="paper",
    yref="paper",
    x=0.5,
    y=1.1,
    text=sfc.string_of_config(config),
    showarrow=False,
    font=dict(
      size=14
    )
  )


  fig.write_image(f'{sfc.GRAPH_DIR}/flame_graph.png', width=1920, height=1080)
  