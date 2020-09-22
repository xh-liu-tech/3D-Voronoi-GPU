#ifndef H_VORONOI_DEFS_H
#define H_VORONOI_DEFS_H

enum Status {
    triangle_overflow = 0,
    vertex_overflow = 1,
    inconsistent_boundary = 2,
    security_radius_not_reached = 3,
    success = 4,
    needs_exact_predicates = 5,
    no_intersection = 6
};

#endif
