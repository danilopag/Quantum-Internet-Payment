from argparse import ArgumentParser
import matplotlib.pyplot as plt
import pandas
import numpy as np
import os

from netsquid_simulationtools.process_qkd import _estimate_bb84_secret_key_rate


def plot_qkd_data(filename, scan_param_name, scan_param_label, save_filename=None, shaded=True, skr_legend=None,
                  skr_minmax=False, plot_skr_qber_dur=(True, True, True), show_fit_lines=(True, True, True),
                  skr_ylim=None, qber_ylim=None, att_ylim=None, normalization=None, convert_to_per_second=False,
                  save_formats=(".eps", ".svg", ".pdf", ".png")):
    """Read a combined container with processed QKD data and plots secret-key rate, QBERs and generation duration per
    success over the specified/varied parameter.

    Parameters
    ----------
    filename : str
        Filename of a csv file containing the combined processed QKD data and the varied parameter. This .csv file can
        be generated with :meth:`process_qkd_data` in `repchain_data_process.py`. The .csv file has to have the
        following columns: scan_param_name, 'sk_rate', 'sk_rate_lower_bound', 'sk_rate_upper_bound', 'Qber_x', 'Qber_z',
        and 'duration_per_success'.
    scan_param_name : str
        Name of parameter that was varied over and should be plotted on the x-axis.
    scan_param_label : str
        Label of the parameter that was varied and should be plotted on the x-axis.
        For example if length was the varied parameter, its label would be `Total Length [km]`.
    save_filename : str (optional)
        Name of the file the figure should be saved as.
        Default is None which creates a name for the saved plot using the name of the input file (filename).
    shaded : Boolean (optional)
        Whether the SKR data should have shaded visualisation of the error.
    skr_legend : list of str (optional)
        Sets a legend for the SKR plot.
    skr_minmax:
        Whether the SKR plot should show regular error bars or min/max.
    plot_skr_qber_dur : tuple of 3 Booleans (optional)
        Specifies which plots should be plotted (Secret Key Rate, QBER, duration per success).
        By default, all three plots are plotted.
    show_fit_lines : tuple of 3 Booleans (optional)
        Specifies whether lines of fit should be plotted for the respective plots (SKR, QBER, duration per success).
    skr_ylim : tuple of 2 floats (optional)
        Sets the limits for y-axis in the Secret Key Rate plot.
        When None, no limits are specified (default).
    qber_ylim : tuple of 2 floats (optional)
        Sets the limits for y-axis in the QBER plot.
        When None, no limits are specified (default).
    att_ylim : tuple of 2 floats (optional)
        Sets the limits for y-axis in the duraiton per success plot.
        When None, no limits are specified (default).
    normalization : float  or None (optional)
        Normalization constant to calculate the normalized secret key rate. Can not be set to 0.
        If None, normalized Secret key rate will not be plotted.
    convert_to_per_second : Boolean (optional)
        Whether to convert the unit of SKR from bits per attempt into bits per second.
    save_formats : String or tuple of Strings
        Defines the file formats in which the resulting plot should be saved.

    Returns
    -------
    plt : `matplotlib.pyplot`
        Object with resulting plot.
    """
    if normalization == 0:
        raise ValueError("Normalization can not be zero. Division by 0 not allowed.")
    elif normalization is None:
        norm = 1
    else:
        norm = normalization

    df_sk = pandas.read_csv(filename)
    df_sk = df_sk.sort_values(by=[scan_param_name])

    # fit
    skr_fit, norm_fit, qber_x_fit, qber_z_fit = fit_skr(df_sk, scan_param_name=scan_param_name,
                                                        normalization=norm)
    # calculate normalized SKR and errors
    df_sk["norm_sk_rate"] = df_sk["sk_rate"] / norm
    df_sk["norm_sk_error"] = df_sk["sk_error"] / norm
    df_sk["norm_sk_rate_lower_bound"] = df_sk["sk_rate_lower_bound"] / norm
    df_sk["norm_sk_rate_upper_bound"] = df_sk["sk_rate_upper_bound"] / norm

    # pick kind of error bars, minmax or symmetric stdev
    if skr_minmax:
        skr_error = [df_sk.sk_rate - df_sk.sk_rate_lower_bound, df_sk.sk_rate_upper_bound - df_sk.sk_rate]
        norm_skr_error = [df_sk.norm_sk_rate - df_sk.norm_sk_rate_lower_bound,
                          df_sk.norm_sk_rate_upper_bound - df_sk.norm_sk_rate]
    else:
        skr_error = df_sk.sk_error
        norm_skr_error = df_sk.norm_sk_error

    # Create plots
    num_plots = plot_skr_qber_dur.count(True)
    _set_font_sizes()
    fig, ax = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))

    if plot_skr_qber_dur[0]:
        # Get axis for SKR plot
        skr_ax = ax if num_plots == 1 else ax[0]

        # Plot secret key rate without normalization
        if normalization is None:

            if convert_to_per_second:
                source_frequency = int(df_sk.source_frequency[0])
                df_sk.sk_rate *= source_frequency
                df_sk.sk_rate_lower_bound *= source_frequency
                df_sk.sk_rate_upper_bound *= source_frequency
                skr_fit_converted = []
                for val in skr_fit:
                    skr_fit_converted.append(val * source_frequency)
                skr_fit = skr_fit_converted
                print("Converted secret key rate from bits per attempt to bits per second.")

                # Save fit data in separate file
                # fit_output = pandas.DataFrame(data={
                #     "length": df_sk.length,
                #     "skr_fit": skr_fit_converted,
                #     "skr_lower": df_sk.sk_rate - df_sk.sk_rate_lower_bound,
                #     "skr_upper": df_sk.sk_rate_upper_bound - df_sk.sk_rate})
                # fit_output.to_csv("fit_output.csv", index=False)

            if shaded:
                # Plot with shaded area
                df_sk.plot(x=scan_param_name, y="sk_rate", yerr=skr_error, kind="scatter", color="xkcd:orange",
                           ax=skr_ax, logy=True, grid=True)

                # Plot normalized SKR as well
                # df_sk.plot(x=scan_param_name, y="norm_sk_rate", kind="scatter", color="blue", ax=skr_ax,
                # logy=True, grid=True)

                # skr_ax.plot(df_sk.get(scan_param_name), norm_fit, 'b')

                skr_ax.fill_between(df_sk.get(scan_param_name), df_sk.sk_rate_lower_bound, df_sk.sk_rate_upper_bound, alpha=0.2,
                                    edgecolor="xkcd:orange", facecolor="xkcd:orange", linewidth=4, antialiased=True)

            else:
                df_sk.plot(x=scan_param_name, y="sk_rate", yerr=skr_error, kind="scatter",
                           color="xkcd:orange", ax=skr_ax, logy=False, grid=True)

            if show_fit_lines[0]:
                skr_ax.plot(df_sk.get(scan_param_name), skr_fit, 'r')

            if convert_to_per_second:
                skr_ax.set_ylabel("Secret key rate [bits/s]")
            elif df_sk.generation_duration_unit[0] == "seconds":
                skr_ax.set_ylabel("Secret key rate [bits/s]")
            elif df_sk.generation_duration_unit[0] == "rounds":
                skr_ax.set_ylabel("Secret key rate [bits/att.]")
            else:
                skr_ax.set_ylabel(f"Secret key rate [bits/{df_sk.generation_duration_unit[0]}]")

            if skr_legend is not None:
                skr_ax.legend(skr_legend)

        # Plot secret key rate with normalization
        else:
            df_sk.plot(x=scan_param_name, y="norm_sk_rate", yerr=norm_skr_error, capsize=4, kind="line", color="red",
                       ax=skr_ax, logy=True)
            if "plob_bound" in df_sk.columns:
                # Plot capacity bounds
                df_sk.plot(x=scan_param_name, y="plob_bound", kind="line", color="blue", ax=skr_ax, logy=True)
                df_sk.plot(x=scan_param_name, y="tgw_bound", kind="line", color="green", ax=skr_ax, logy=True)
                skr_ax.legend(["PLOB (blue)", "TGW (green)", "Normalized Rate (red)"])

            skr_ax.set_ylabel("Normalized Secret-Key Rate [bits/attempt]")

        if skr_ylim is not None:
            skr_ax.set_ylim(skr_ylim)

        skr_ax.set_xlabel(scan_param_label)

    # Plot QBER (with fit)
    if plot_skr_qber_dur[1]:
        # Get axis for QBER plot
        qber_ax = ax if num_plots == 1 else ax[plot_skr_qber_dur[:-1].count(True) - 1]

        df_sk.plot(x=scan_param_name, y="Qber_x", yerr="Qber_x_error", kind="scatter", color="red", ax=qber_ax,
                   logy=False, grid=True)
        df_sk.plot(x=scan_param_name, y="Qber_z", yerr="Qber_z_error", kind="scatter", color="blue", ax=qber_ax,
                   logy=False, grid=True)

        if show_fit_lines[1]:
            qber_ax.plot(df_sk.get(scan_param_name), qber_z_fit, 'b')
            qber_ax.plot(df_sk.get(scan_param_name), qber_x_fit, 'r')
            qber_ax.legend(["Z basis (fit)", "X basis (fit)", "X basis", "Z basis"])
        else:
            qber_ax.legend(["X basis", "Z basis"])

        qber_ax.set_ylabel("QBER")

        if qber_ylim is not None:
            qber_ax.set_ylim(qber_ylim)

        qber_ax.set_xlabel(scan_param_label)

    # Plot generation duration per success
    if plot_skr_qber_dur[2]:
        # Get axis for generation duration per success plot
        att_ax = ax if num_plots == 1 else ax[plot_skr_qber_dur.count(True) - 1]
        plot_duration(ax=att_ax, dataframe=df_sk, scan_param_name=scan_param_name, scan_param_label=scan_param_label,
                      show_fit_line=show_fit_lines[2], ylim=att_ylim)

    # try to make sure there is enough spacing between subplots
    fig.tight_layout()

    # Save figure
    save_filename = save_filename if save_filename is not None else "plot_qkd_data_" + filename.split("/")[-1][:-4]
    if type(save_formats) is str:
        fig.savefig(save_filename + save_formats, bbox_inches="tight")
    else:
        for fileformat in save_formats:
            fig.savefig(save_filename + fileformat, bbox_inches="tight")

    plt.show()
    return plt


def plot_teleportation(filename, scan_param_name, scan_param_label, save_filename=None, show_duration=True,
                       show_average_fidelity=True, show_minimum_fidelity_xyz=True,
                       show_average_fidelity_optimized_local_unitaries=True,
                       save_formats=(".eps", ".svg", ".pdf", ".png")):
    """Plot processed data with quantum-teleportation fidelity.
       Parameters
    ----------
    filename : str
        Filename of a csv file containing the processed teleportation data and the varied parameter. This .csv file can
        be generated with :func:`process_data` in `repchain_data_process.py`.
    scan_param_name : str
        Name of parameter that was varied over and should be plotted on the x-axis.
    scan_param_label : str
        Label of the parameter that was varied and should be plotted on the x-axis.
        For example if length was the varied parameter, its label would be `Total Length [km]`.
    save_filename : str (optional)
        Name of the file the figure should be saved as.
        Default is None which creates a name for the saved plot using the name of the input file (filename).
    show_duration : bool (optional)
        If true, the duration per distributed entangled pair is plotted.
        Requires the .csv file to have the columns "duration_per_success" and "duration_per_success_error".
    show_average_fidelity : bool (optional)
        If true, the average teleportation fidelity is plotted.
        Requires the .csv file to have the columns "teleportation_fidelity_average" and
        "teleportation_fidelity_average_error".
    show_minimum_fidelity_xyz : bool (optional)
        If true, the teleportation fidelity minimized over all eigenstates of the Pauli X, Y and Z operators is plotted.
        Requires the .csv file to have the columns "teleportation_fidelity_minimum_xyz_eigenstates" and
        "teleportation_fidelity_minimum_xyz_eigenstates_error".
    show_average_fidelity_opzimized_local_unitaries: bool (optional)
        If true, the average teleportation fidelity maximized over local unitary operations is plotted.
        Requires the .csv file to have the column "teleportation_fidelity_average_optimized_local_unitaries".
    save_formats : String or tuple of Strings
        Defines the file formats in which the resulting plot should be saved.

    Returns
    -------
    plt : `matplotlib.pyplot`
        Object with resulting plot.

    """

    df = pandas.read_csv(filename)
    df = df.sort_values(by=[scan_param_name])

    num_plots = 2 if show_duration else 1
    _set_font_sizes(medium=14)
    fig, ax = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))

    fidelity_ax = ax if num_plots == 1 else ax[0]
    fidelity_ax.set_xlabel(scan_param_label)
    fidelity_ax.set_ylabel("Teleportation Fidelity")
    if show_average_fidelity:
        df.plot(x=scan_param_name, y="teleportation_fidelity_average",
                yerr="teleportation_fidelity_average_error", alpha=1, zorder=1,
                capsize=4, kind="line", color="red", label="average",
                ax=fidelity_ax)
    if show_minimum_fidelity_xyz:
        df.plot(x=scan_param_name, y="teleportation_fidelity_minimum_xyz_eigenstates",
                yerr="teleportation_fidelity_minimum_xyz_eigenstates_error", capsize=4, kind="line", zorder=2,
                style="y--", label="minimum over XYZ eigenstates", ax=fidelity_ax)
    if show_average_fidelity_optimized_local_unitaries:
        df.plot(x=scan_param_name, y="teleportation_fidelity_average_optimized_local_unitaries", kind="line",
                style="g:", linewidth=3, zorder=3,
                label="average optimized over local unitaries", ax=fidelity_ax)
    fidelity_ax.legend()

    if show_duration:
        plot_duration(ax=ax[1], dataframe=df, scan_param_name=scan_param_name, scan_param_label=scan_param_label,
                      show_fit_line=True, ylim=None)

    # try to make sure there is enough spacing between subplots
    fig.tight_layout()

    # Save figure
    save_filename = save_filename if save_filename is not None else "plot_teleportation_data_" + \
                                                                    filename.split("/")[-1][:-4]
    if type(save_formats) is str:
        fig.savefig(save_filename + save_formats, bbox_inches="tight")
    else:
        for fileformat in save_formats:
            fig.savefig(save_filename + fileformat, bbox_inches="tight")

    plt.show()
    return plt


def plot_fidelity_rate(filename, scan_param_name, scan_param_label, save_filename=None, plot_fid_rate=(True, True),
                       show_fit_lines=(True, True)):
    """Read a combined container with processed data and plots fidelity and generation duration over the
    specified/varied parameter.

    Parameters
    ----------
    filename: str
        Filename of a csv file containing the processed data and the varied parameter. This .csv file can
        be generated with :meth:`process_fidelity_data` in `repchain_data_process.py`. The .csv file has to have the
        following columns: scan_param_name, 'fidelity' and 'fidelity_error' (if plotting fidelity),
        'duration_per_success' and 'duration_per_success_error' (if plotting rate).
    scan_param_name: str
        Name of parameter that was varied over and should be plotted on the x-axis.
    scan_param_label : str
        Label (name and unit) of the parameter that was varied and should be plotted on the x-axis.
        For example if length was the varied parameter, its label would be `Total Length [km]`.
    save_filename : str (optional)
        Name of the file the figure should be saved as.
        Default is None which creates a name for the saved plot using the name of the input file (filename).
    plot_fid_rate : tuple of 2 Booleans (optional)
        Specifies which plots should be plotted (fidelity, rate).
        By default, both plots are plotted.
    show_fit_lines : tuple of 2 Booleans (optional)
        Specifies whether lines of fit should be plotted for the respective plots (fidelity, rate).

    Returns
    -------
    plt : `matplotlib.pyplot`
        Object with resulting plot.
    """
    df_fid = pandas.read_csv(filename)

    # Create plots
    num_plots = plot_fid_rate.count(True)
    _set_font_sizes()
    fig, ax = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))

    # Plot fidelity
    if plot_fid_rate[0]:
        # Get axis for fidelity graph
        fid_ax = ax if num_plots == 1 else ax[0]

        df_fid.plot(x=scan_param_name, y="fidelity", yerr="fidelity_error", kind="scatter", color="blue", ax=fid_ax,
                    logy=True, grid=True)

        if show_fit_lines[0]:
            fit = np.poly1d(np.polyfit(df_fid.get(scan_param_name), df_fid.fidelity, 1))
            fid_fit = []
            for k in range(len(df_fid.get(scan_param_name))):
                fid_fit.append(fit(df_fid.get(scan_param_name)[k]))
            fid_ax.plot(df_fid.get(scan_param_name), fid_fit, 'b')

        fid_ax.set_ylabel("Average Fidelity")

        fid_ax.set_xlabel(scan_param_label)

    # Plot rate
    if plot_fid_rate[1]:
        # Get axis for rate graph
        rate_ax = ax if num_plots == 1 else ax[1]

        df_fid.plot(x=scan_param_name, y="duration_per_success", kind="scatter", color="red",
                    yerr="duration_per_success_error", ax=rate_ax, legend=False, grid=True)

        if show_fit_lines[1]:
            fit = np.poly1d(np.polyfit(df_fid.get(scan_param_name), df_fid.duration_per_success, 1))
            rate_fit = []
            for k in range(len(df_fid.get(scan_param_name))):
                rate_fit.append(fit(df_fid.get(scan_param_name)[k]))
            rate_ax.plot(df_fid.get(scan_param_name), rate_fit, 'r')

        rate_ax.set_ylabel("Average generation duration")

        rate_ax.set_xlabel(scan_param_label)

    # try to make sure there is enough spacing between subplots
    fig.tight_layout()

    save_filename = save_filename if save_filename is not None else "plot_fidelity_" + filename.split("/")[-1][:-4]
    plt.savefig(save_filename)

    plt.show()
    return plt


def plot_multiple_qkd(raw_data_dir=".", save_filename="multiple_qkd_plot", plot_skr_qber_dur=(True, True, True),
                      convert_to_per_second=False, show_fit_lines=(True, True, True), skr_ylim=None, qber_ylim=None,
                      dur_ylim=None, shaded=False):
    """Plot secret-key rate, QBERs and generation duration per success over the specified/varied parameter
    for different QKD experiments in one plot.

    Parameters
    ----------
    raw_data_dir : str (optional)
        Directory with data that should be plotted. The .csv files with data to be plotted should contain columns
        'length', 'sk_rate', 'sk_rate_lower_bound', 'sk_rate_upper_bound', 'Qber_x', 'Qber_z', and
        'duration_per_success'.
    save_filename : str (optional)
        Name of the file the figure should be saved as.
    plot_skr_qber_dur : tuple of 3 Booleans (optional)
        Specifies which plots should be plotted (Secret Key Rate, QBER, duration per success).
        By default, all three plots are plotted.
    convert_to_per_second : Boolean (optional)
        Whether to convert the unit of SKR from bits per attempt into bits per second.
    show_fit_lines : tuple of 3 Booleans (optional)
        Specifies whether lines of fit should be plotted for the respective plots (SKR, QBER, duration per success).
    skr_ylim : tuple of 2 floats (optional)
        Sets the limits for y-axis in the Secret Key Rate plot.
        When None, no limits are specified (default).
    qber_ylim : tuple of 2 floats (optional)
        Sets the limits for y-axis in the QBER plot.
        When None, no limits are specified (default).
    dur_ylim : tuple of 2 floats (optional)
        Sets the limits for y-axis in the duration per success plot.
        When None, no limits are specified (default).
    shaded : bool
        Whether the SKR plot should display error bars as shaded or regular.

    Returns
    -------
    plt : `matplotlib.pyplot`
        Object with resulting plot.
    """

    if not os.path.exists(raw_data_dir):
        raise NotADirectoryError("No raw_data directory found!")

    skr_legend = []
    qber_legend = []
    att_legend = []
    # Create plots
    num_plots = plot_skr_qber_dur.count(True)
    _set_font_sizes()
    fig, ax = plt.subplots(1, num_plots, figsize=(7 * num_plots, 6))
    i = 0
    for k, filename in enumerate(os.listdir(raw_data_dir)):
        if filename[-3:] == "csv":
            i += 1

            # Read out csv files
            csv_data = pandas.read_csv(raw_data_dir + "/" + filename)

            skr_fit, norm_fit, qber_x_fit, qber_z_fit = fit_skr(csv_data, scan_param_name="length")

            # convert SKR to account for forced measurement basis
            csv_data.sk_rate /= 2
            csv_data.sk_error /= 2
            skr_fit = [x / 2 for x in skr_fit]
            norm_fit = [x / 2 for x in norm_fit]

            if plot_skr_qber_dur[0]:
                # Get axis for SKR plot
                skr_ax = ax if num_plots == 1 else ax[0]
                skr_legend.append(filename[:-4])

                # Convert skr from [bits/attempt] to [bits/s]
                if convert_to_per_second:
                    source_frequency = int(csv_data.source_frequency[0])
                    csv_data.sk_rate *= source_frequency
                    csv_data.sk_rate_lower_bound *= source_frequency
                    csv_data.sk_rate_upper_bound *= source_frequency
                    skr_fit_converted = []
                    for val in skr_fit:
                        skr_fit_converted.append(val * source_frequency)

                    if show_fit_lines[0]:
                        skr_ax.plot(csv_data.length, skr_fit_converted, "C" + str(int(i/2)))
                        skr_legend.append(filename[:-4] + " (fit)")

                if shaded:
                    skr_ax.fill_between(csv_data.length, csv_data.sk_rate - csv_data.sk_error,
                                        csv_data.sk_rate + csv_data.sk_error,
                                        alpha=.15, edgecolor="C" + str(int(i/2)), facecolor="C" + str(int(i/2)),
                                        linewidth=1, antialiased=True)
                else:
                    skr_ax.errorbar(csv_data.length, csv_data.sk_rate, yerr=csv_data.sk_error,
                                    color="C" + str(int(i/2)), alpha=0.5, ls='')

                skr_ax.plot(csv_data.length, csv_data.sk_rate, "o", color="C" + str(int(i/2)))

                if show_fit_lines[0] and not convert_to_per_second:
                    skr_ax.plot(csv_data.length, skr_fit, "C" + str(int(i/2)))
                    # skr_legend.append(filename[:-4] + " (fit)")
                    skr_legend.append('_nolegend_')

                skr_ax.set_ylabel("Secret Key Rate [bits/s]" if convert_to_per_second else "Secret Key Rate [bits/att.]")

                # modified symmetrical logscale with variable linear interval
                skr_ax.set_yscale("symlog", linthreshy=0.5*1e-5)

                if skr_ylim is not None:
                    skr_ax.set_ylim(skr_ylim)

                skr_ax.legend(skr_legend, frameon=False)

            # Plot QBER (with fit)
            if plot_skr_qber_dur[1]:
                # Get axis for QBER plot
                qber_ax = ax if num_plots == 1 else ax[plot_skr_qber_dur[:-1].count(True) - 1]

                qber_ax.errorbar(csv_data.length, csv_data.Qber_x, yerr=csv_data.Qber_x_error, color="C" + str(2*i-1),
                                 alpha=0.5, ls='')
                qber_ax.plot(csv_data.length, csv_data.Qber_x, "x", color="C" + str(2*i-1))
                qber_ax.errorbar(csv_data.length, csv_data.Qber_z, yerr=csv_data.Qber_z_error, color="C" + str(2*i),
                                 alpha=0.5, ls='')
                qber_ax.plot(csv_data.length, csv_data.Qber_z, "x", color="C" + str((2*i)))

                if show_fit_lines[1]:
                    qber_ax.plot(csv_data.length, qber_z_fit, 'b', color="C" + str(2*i))
                    qber_ax.plot(csv_data.length, qber_x_fit, 'r', color="C" + str(2*i-1))
                    qber_legend.append(filename[:-4] + " X basis")
                    qber_legend.append(filename[:-4] + " Z basis")
                    # qber_legend.append(filename[:-4] + "_X (fit)")
                    # qber_legend.append(filename[:-4] + "_Z (fit)")
                    qber_legend.append('_nolegend_')
                    qber_legend.append('_nolegend_')
                else:
                    qber_legend.append(filename[:-4] + "_X basis")
                    qber_legend.append(filename[:-4] + "_Z basis")

                if qber_ylim is not None:
                    qber_ax.set_ylim(qber_ylim)

                qber_ax.set_ylabel("QBER")
                qber_ax.locator_params(axis='y', nbins=4)

                qber_ax.legend(qber_legend, frameon=False)

            # Plot duration per success
            if plot_skr_qber_dur[2]:
                # Get axis for duration per success plot
                att_ax = ax if num_plots == 1 else ax[plot_skr_qber_dur.count(True) - 1]

                att_ax.errorbar(csv_data.length, csv_data.duration_per_success,
                                yerr=csv_data.duration_per_success_error, color="C" + str(i), alpha=0.5, ls='')
                att_ax.plot(csv_data.length, csv_data.duration_per_success, "o")

                if show_fit_lines[2]:
                    fit = np.poly1d(np.polyfit(csv_data.length, np.log(csv_data.duration_per_success), deg=5))
                    att_fit = []
                    for k in range(len(csv_data.length)):
                        att_fit.append(np.exp(fit(csv_data.length[k])))
                    att_ax.plot(csv_data.length, att_fit, 'b')

                att_ax.set_ylabel("Duration per success")
                att_ax.set_yscale("log")

                att_legend.append(filename[:-4])

                if dur_ylim is not None:
                    att_ax.set_ylim(dur_ylim)

                att_ax.legend(att_legend, frameon=False)

    for a in ax:
        a.set_xlabel("Total Distance [km]")
        a.set_xlim(0, 115)
        a.grid()

    # try to make sure there is enough spacing between subplots
    fig.tight_layout()

    # Save figure
    save_formats = (".svg", ".pdf", ".png")
    for fileformat in save_formats:
        fig.savefig(save_filename + fileformat, bbox_inches="tight")

    plt.show()
    return plt


def plot_multiple_fidelity(raw_data_dir=".", show_fit_line=True, ylim=None):
    """Plot Fidelity for different QKD experiments in one plot.

    Parameters
    ----------
    raw_data_dir : str (optional)
        Directory with data that should be plotted. The .csv files with data to be plotted should contain columns
        'length', 'fidelity', and 'fidelity_error'.
    show_fit_line : Boolean (optional)
        Whether to show the lines of fit for fidelities.
    ylim : tuple of 2 floats (optional)
        Sets the limits for y-axis in the plot.
        When None, no limits are specified (default).

    Returns
    -------
    plt : `matplotlib.pyplot`
        Object with resulting plot.
    """

    if not os.path.exists(raw_data_dir):
        raise NotADirectoryError("No raw_data directory found!")

    legend = []
    _set_font_sizes()
    fig, ax = plt.subplots(1, 1)

    for i, filename in enumerate(os.listdir(raw_data_dir)):
        if filename[-3:] == "csv":
            legend.append(filename[:-4])

            # Read out csv files
            csv_data = pandas.read_csv(raw_data_dir + "/" + filename)

            ax.errorbar(csv_data.length, csv_data.fidelity, yerr=csv_data.fidelity_error, color="C" + str(i), alpha=0.5, ls='')
            ax.plot(csv_data.length, csv_data.fidelity, "o")

            if show_fit_line:
                fit = np.poly1d(np.polyfit(csv_data.length, csv_data.fidelity, 1))
                fid_fit = []
                for k in range(len(csv_data.length)):
                    fid_fit.append(fit(csv_data.length[k]))
                ax.plot(csv_data.length, fid_fit, "C" + str(i))
                legend.append(filename[:-4] + " (fit)")

    ax.set_xlabel("Total Length [km]")
    ax.legend(legend, frameon=False)

    ax.set_ylabel("Average Fidelity")
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_yscale("log")

    # try to make sure there is enough spacing between subplots
    fig.tight_layout()

    plt.show()
    return plt


def fit_skr(df_sk, scan_param_name, deg=1, normalization=1.):
    """Fit polynomial of degree deg to QBER and calculate Secret key rate from fitted data.

    Parameters
    ----------
    df_sk : pandas.dataframe
        Dataframe containing the secret key rate data.
    scan_param_name : str
        Name of parameter that SKR should be fitted against.
    deg : int (optional)
        Degree of the polynomial to be fitted. Default is a linear fit.
    normalization : float (optional)
        Normalization constant to calculate the normalized secret key rate.

    Returns
    -------
    skr_fit : list
        List with values of secret key rate fitted for each datapoint of QBER.
    norm_fit : list
        List with values of normalized secret key rate fitted for each datapoint of QBER. If no normalization was
        specified this is an empty list.
    qber_x_fit : list
        List with values of fitted QBER in X basis.
    qber_z_fit : list
        List with values of fitted QBER in Z basis.
    """
    if normalization == 0:
        raise ValueError("Normalization can not be zero. Division by 0 not allowed.")

    skr_fit = []
    norm_fit = []

    # numpy polynomial fit

    fit_x = np.poly1d(np.polyfit(df_sk.get(scan_param_name), df_sk.Qber_x, deg=deg))
    fit_z = np.poly1d(np.polyfit(df_sk.get(scan_param_name), df_sk.Qber_z, deg=deg))

    qber_x_fit = []
    qber_z_fit = []
    for k in range(len(df_sk.get(scan_param_name))):
        qber_x_fit.append(fit_x(df_sk.get(scan_param_name)[k]))
        qber_z_fit.append(fit_z(df_sk.get(scan_param_name)[k]))

    for n in range(len(qber_x_fit)):
        secret_key_rate_fit, skr_min_fit, skr_max_fit, _ = _estimate_bb84_secret_key_rate(qber_x_fit[n], 0, qber_z_fit[n], 0,
                                                                                          df_sk.duration_per_success[n],
                                                                                          df_sk.duration_per_success_error[n])
        skr_fit.append(secret_key_rate_fit)
        norm_fit.append(secret_key_rate_fit / normalization)

    return skr_fit, norm_fit, qber_x_fit, qber_z_fit


def plot_duration(ax, dataframe, scan_param_name, scan_param_label, show_fit_line, ylim):
    # Get axis for generation duration per success plot

    dataframe.plot(x=scan_param_name, y="duration_per_success", kind="scatter", color='green',
                   yerr="duration_per_success_error", ax=ax, legend=True, logy=True, grid=True)

    if show_fit_line:
        fit = np.poly1d(np.polyfit(dataframe.get(scan_param_name), np.log(dataframe.duration_per_success), deg=5))
        att_fit = []
        for k in range(len(dataframe.get(scan_param_name))):
            att_fit.append(np.exp(fit(dataframe.get(scan_param_name)[k])))
        ax.plot(dataframe.get(scan_param_name), att_fit, 'b')

    if dataframe.generation_duration_unit[0] == "seconds":
        ax.set_ylabel("Avg. num. of s/succ.")
    elif dataframe.generation_duration_unit[0] == "rounds":
        ax.set_ylabel("Avg. num. of att./succ.")
    else:
        ax.set_ylabel(f"Avg. num. of {dataframe.generation_duration_unit[0]}/succ.")

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xlabel(scan_param_label)


def _set_font_sizes(small=17, medium=20, large=24, usetex=False):
    """Set font sizes of the plot.

    Parameters
    ----------
    small : float (optional)
        Size of default font, axes labels, ticks and legend.
    medium : float (optional)
        Fontsize of x and y labels.
    large : float (optional)
        Fontsize of figure title.
    usetex : bool (optional)
        Whether to use latex for all text output. Default is False.
    """

    plt.rc('font', size=small)          # controls default text sizes
    plt.rc('axes', titlesize=medium)     # fontsize of the axes title
    plt.rc('axes', labelsize=large)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=small)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=small)    # fontsize of the tick labels
    plt.rc('legend', fontsize=medium)    # legend fontsize
    plt.rc('figure', titlesize=large)   # fontsize of the figure title
    plt.rc('text', usetex=usetex)         # use LaTeX for text output


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('-f', '--filename', required=False, type=str, default='output.csv',
                        help="Name of CSV file (output of repchain_data_process.py)")
    parser.add_argument('-p', '--parameter', required=True, type=str,
                        help="Name of the varied parameter.")
    parser.add_argument('-m', '--mode', type=str, required=False, choices=['teleportation', 'bb84'],
                        default='teleportation')
    args = parser.parse_args()
    if args.mode == 'teleportation':
        plot_teleportation(filename=args.filename, scan_param_name=args.parameter, scan_param_label=args.parameter)
    elif args.mode == 'bb84':
        plot_qkd_data(filename=args.filename, scan_param_name=args.parameter, scan_param_label=args.parameter)
