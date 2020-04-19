# ### Characterize random access to storagex
pp = PdfPages('MemoryBlockFigure.pdf')
figure(figsize=(6,4))

Colors='bgrcmyk'  # The colors for the plot
LineStyles=['-']

fig = matplotlib.pyplot.gcf()
fig.set_size_inches(18.5,10.5)

for m_i in range(len(m_list)):
    TMem[m_i],_mean[m_i],_std[m_i],m_legend[m_i]
    Color=Colors[m_i % len(Colors)]
    PlotTime(TMem[m_i],_mean[m_i],_std[m_i],             Color=Color,LS='-',Legend=m_legend[m_i],             m_i=m_i)

grid()
legend(fontsize=18)
xlabel('delay (sec)',fontsize=18)
ylabel('1-CDF',fontsize=18)
tick_params(axis='both', which='major', labelsize=16)
tick_params(axis='both', which='minor', labelsize=12)
pp.savefig()
pp.close()