"""
function contour_heatmap!(ax, ϕ, p_coord, slice_zonal, contour_levels, colorrange; 
    add_labels = false, title_string = "", colormap = :balance,
    heuristic = 1, random_seed = 12345)
# Description 
Combines a contour plot with a heatmap and adds numbers to the contours.
"""
function contour_heatmap!(ax, ϕ, p_coord, slice_zonal, contour_levels, colorrange; colormap=:balance, labelsize = 40, markersize = 25)
# add_labels=false,  heuristic=1, random_seed=12345, labelsize = 40, markersize = 25)
    
    hm = heatmap!(ax, ϕ, p_coord, slice_zonal; levels=contour_levels, interpolate = true, colorrange=colorrange, colormap=colormap, labelsize = labelsize)
    cplot = contour!(ax, ϕ, p_coord, slice_zonal; levels=contour_levels, color=:black, labelsize = markersize, labels = true )

    ax.limits = (extrema(ϕ)..., extrema(p_coord)...)

    ax.xlabel = "Latitude [ᵒ]"
    ax.ylabel = "Stretched Height"
    ax.xlabelsize = labelsize
    ax.ylabelsize = labelsize
    ax.xticklabelsize = labelsize
    ax.yticklabelsize = labelsize

    ax.xticks = ([-75, -50, -25, 0, 25, 50, 75], ["75S", "50S", "25S", "0", "25N", "50N", "75N"])
    pressure_levels = [1000, 850, 700, 550, 400, 250, 100, 10]
    ax.yticks = (pressure_levels .* 1e2, string.(pressure_levels))
    ax.yreversed = true

    # hack 
    #=
    Random.seed!(random_seed)
    if add_labels
        list_o_stuff = []
        labeled_contours = contour_levels[1:1:end]
        for level in labeled_contours
            local fig_t, ax_t, cp_t = contour(ϕ, p_coord, slice_zonal, levels=[level], linewidth=0)
            local segments = cp_t.plots[1][1][]
            local index_vals = []
            local beginnings = []
            for (i, p) in enumerate(segments)
                # the segments are separated by NaN, which signals that a new contour starts
                if isnan(p)
                    push!(beginnings, segments[i-1])
                    push!(index_vals, i)
                end
            end
            push!(list_o_stuff, (; segments, beginnings, index_vals))
        end

        for contour_index = 1:length(labeled_contours)

            local contour_val = labeled_contours[contour_index]
            local segments = list_o_stuff[contour_index].segments

            local indices = [0, list_o_stuff[contour_index].index_vals[1:end]...]
            for i = 1:length(indices)-1
                # heuristics for choosing where on line
                local index1 = rand(indices[i]+1:indices[i+1]-1) # choose random point on segment
                local index2 = round(Int, 0.5 * indices[i] + 0.5 * indices[i+1]) # choose point in middle
                β = (rand() - 0.5) * 0.9 + 0.5 # α ∈ [0,1]
                # α = (contour_index-1) / (length(labeled_contours)-1)
                α = contour_index % 2 == 0 ? 0.15 : 0.85
                α = rand([α, β])
                local index3 = round(Int, α * (indices[i] + 1) + (1 - α) * (indices[i+1] - 1)) # choose point in middle
                if heuristic == 3
                    local index = index3 # rand([index1, index2]) # choose between random point and point in middle
                elseif heuristic == 1
                    local index = index1
                elseif heuristic == 2
                    local index = index2
                end
                # end of heuristics
                local location = Point3(segments[index]..., 2.0f0)
                local sc = scatter!(ax, location, markersize=markersize , align=(:center, :center), color=(:white, 0.1), strokecolor=:white)
                local anno = text!(ax, [("$contour_val", location)], align=(:center, :center), fontsize=markersize , color=:black)

                delete!(ax, sc)
                delete!(ax, cplot)
                delete!(ax, anno)

                push!(ax.scene, anno)
                push!(ax.scene, sc)
                push!(ax.scene, cplot)
            end
        end
    end # end of adding labels
    =#
end