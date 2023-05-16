# function D3Trees.D3Tree(policy::CMCTSPlanner, root_state; kwargs...)
#     # check to see if visualization was enabled
#     @warn("""
#          D3Tree(planner::CMCTSPlanner, root_state) is deprecated and may be removed in the future. Instead, please use
# 
#              a, info = action_info(planner, state)
#              D3Tree(info[:tree])
# 
#          Make sure that the tree_in_info solver option is set to true. You can also get this info from a POMDPToolbox History
# 
#              info = first(ainfo_hist(hist))
#              D3Tree(info[:tree])
#          """)
#     if !policy.solver.enable_tree_vis
#         error("""
#               Tree visualization was not enabled for this policy.
# 
#               Construct the solver with $(typeof(policy.solver))(enable_tree_vis=true, ...) to enable.
#               """)
#     end
#     return D3Tree(policy.tree, root_state; kwargs...)
# end

function D3Trees.D3Tree(policy::CDPWPlanner; kwargs...)
    @warn("""
         D3Tree(planner::CDPWPlanner) is deprecated and may be removed in the future. Instead, please use

             a, info = action_info(planner, state)
             D3Tree(info[:tree])

         Make sure that the tree_in_info solver option is set to true. You can also get this info from a POMDPToolbox History

             info = first(ainfo_hist(hist))
             D3Tree(info[:tree])
         """)
    return D3Tree(policy.tree; kwargs...)
end

# function D3Trees.D3Tree(tree::CMCTSTree, root_state=first(tree.s_labels); title="CMCTS tree", kwargs...)
#     if tree._vis_stats == nothing
#         error("""
#               Visualization was not enabled for this tree.
# 
#               Construct the CMCTS solver with 'enable_tree_vis=true' to enable.
#               """)
#     end
# 
#     vs = tree._vis_stats
# 
#     nsas = length(vs)
#     nsa = length(tree.n)
#     nodes = Vector{Dict{String, Any}}(undef, 1 + nsas + nsa)
# 
#     # root node
#     if haskey(tree.state_map, root_state)
#         root_id = tree.state_map[root_state]
#     else
#         error("Could not find state $root_state in tree for visualization.")
#     end
#     nodes[1] = Dict("type"=>:state,
#                     "child_d3ids"=>[1+nsas+c for c in tree.child_ids[root_id]],
#                     "tag"=>node_tag(root_state),
#                     "tt_tag"=>tooltip_tag(root_state),
#                     "n"=>tree.total_n[root_id],
#                     "total_n"=>tree.total_n[root_id],
#                     "parent_n"=>tree.total_n[root_id]
#                    )
# 
#     # state-action nodes
#     for i in 1:nsa
#         a = tree.a_labels[i]
#         nodes[1+nsas+i] = Dict("type"=>:action,
#                                "child_d3ids"=>Int[],
#                                "tag"=>node_tag(a),
#                                "tt_tag"=>tooltip_tag(a),
#                                "n"=>tree.n[i],
#                                "q"=>tree.q[i],
#                                "c"=>tree.qc[i],
#                               )
#     end
# 
#     # state-action-state nodes
#     for (i,((said,sid),n)) in enumerate(vs)
#         s = tree.s_labels[sid]
#         nodes[1+i] = Dict("type"=>:state,
#                           "child_d3ids"=>[1+nsas+c for c in tree.child_ids[sid]],
#                           "tag"=>node_tag(s),
#                           "tt_tag"=>tooltip_tag(s),
#                           "n"=>n,
#                           "total_n"=>tree.total_n[sid],
#                           "parent_n"=>tree.n[said]
#                          )
#         # add as a child to corresponding sa node
#         push!(nodes[1+nsas+said]["child_d3ids"], 1+i)
# 
#         n = total_n(StateNode(tree, sid))
#         # add parent_n to all children
#         for csan in children(StateNode(tree, sid))
#             csaid = csan.id
#             nodes[1+nsas+csaid]["parent_n"] = n
#         end
#     end
# 
#     for csan in children(StateNode(tree, root_id))
#         csaid = csan.id
#         nodes[1+nsas+csaid]["parent_n"] = total_n(StateNode(tree, root_id))
#     end
# 
#     return D3Tree(nodes; title=title, kwargs...)
# end

function D3Trees.D3Tree(tree::CDPWTree; title="CMCTS-DPW Tree", kwargs...)
    lens = length(tree.total_n)
    lensa = length(tree.n)
    len = lens + lensa
    children = Vector{Vector{Int}}(undef, len)
    text = Vector{String}(undef, len)
    tt = fill("", len)
    style = fill("", len)
    link_style = fill("", len)
    max_q = maximum(tree.q)
    min_q = minimum(tree.q)

    for s in 1:lens
        children[s] = tree.children[s] .+ lens
        text[s] =  @sprintf("""
                            %25s
                            N: %6d
                            """,
                            node_tag(tree.s_labels[s]),
                            tree.total_n[s]
                           )
        tt[s] = """
                $(tooltip_tag(tree.s_labels[s]))
                N: $(tree.total_n[s])
                """
        for sa in tree.children[s]
            w = 20.0*sqrt(tree.n[sa]/tree.total_n[s])
            link_style[sa+lens] = "stroke-width:$(w)px"
        end
    end
    for sa in 1:lensa
        children[sa+lens] = let
            sp_children = if !isempty(tree.unique_transitions)
                last.(filter(((sanode,spnode),) -> sanode == sa, tree.unique_transitions))
            else
                first.(tree.transitions[sa])
            end
            @assert length(sp_children) == tree.n_a_children[sa]
            collect(sp_children)
        end

        text[sa+lens] = @sprintf("""
                                 %25s
                                 Q: %6.2f
                                 N: %6d
                                 """,
                                 node_tag(tree.a_labels[sa]),
                                 tree.q[sa],
                                 tree.n[sa]
                                )
        tt[sa+lens] = """
                      $(tooltip_tag(tree.a_labels[sa]))
                      Q: $(tree.q[sa])
                      N: $(tree.n[sa])
                      C: $(tree.qc[sa])
                      """

        rel_q = (tree.q[sa]-min_q)/(max_q-min_q)
        if isnan(rel_q)
            color = colorant"gray"
        else
            color = weighted_color_mean(rel_q, colorant"green", colorant"red")
        end
        style[sa+lens] = "stroke:#$(hex(color))"
    end
    return D3Tree(children;
                  text=text,
                  tooltip=tt,
                  style=style,
                  link_style=link_style,
                  title=title,
                  kwargs...
                 )
end
