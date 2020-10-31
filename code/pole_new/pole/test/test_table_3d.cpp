//
// Created by adewindt on 6/7/20.
//

#include "catch2/catch.hpp"
#include "match_span.h"
#include "types.h"
#include "table_3d.h"
#include "spancpy.h"
#include "fmt/format.h"
#include <array>


namespace pole::tests {
    SCENARIO("Table3D", "") {
        GIVEN("A filled 3x2x2 Table3D<i32>") {
            Table3D<i32, f64> table(
                    -1, 2, 3,
                    -1, 1, 2,
                    -1, 1, 2,
                    0);

            auto table_span = table.get_span();
            spancpy(table_span, {1, 2, 3,
                                        4, 5, 6,
                                         7,  8,  9,
                                        10, 11, 12});

            WHEN("Read all data by index into the list") {
                std::array<i32, 12> result = {
                        table.get(0,0,0),
                        table.get(1,0,0),
                        table.get(2,0,0),
                        table.get(0,1,0),
                        table.get(1,1,0),
                        table.get(2,1,0),

                        table.get(0,0,1),
                        table.get(1,0,1),
                        table.get(2,0,1),
                        table.get(0,1,1),
                        table.get(1,1,1),
                        table.get(2,1,1),
                };
                THEN("Read elements must match the table memory") {
                    CHECK_THAT(result, SpanEquals<i32>(table_span));
                }
            }

            WHEN("Set a single element by index select.") {
                table.get(0, 0, 0) = 99;
                THEN("Element must be set.") {
                    CHECK(table.get(0, 0, 0) == 99);
                }
            }

            WHEN("Select all elements by value.") {
                std::array<i32, 12> result = {
                        table(-0.5,-0.5,-0.5),
                        table(0.5,-0.5,-0.5),
                        table(1.5,-0.5,-0.5),
                        table(-0.5,0.5,-0.5),
                        table(0.5,0.5,-0.5),
                        table(1.5,0.5,-0.5),

                        table(-0.5,-0.5,0.5),
                        table(0.5,-0.5,0.5),
                        table(1.5,-0.5,0.5),
                        table(-0.5,0.5,0.5),
                        table(0.5,0.5,0.5),
                        table(1.5,0.5,0.5),
                };
                THEN("Read elements must match the table memory") {
                    CHECK_THAT(result, SpanEquals<i32>(table_span));
                }
            }

            WHEN("Set a single element by value select.") {
                table(-0.5,0.5,-0.5) = 99;
                THEN("Element must be set.") {
                    CHECK(table.get(0,1,0) == 99);
                }
            }

            WHEN("Span select by index") {
                CHECK_THAT(table.get(0, 0), SpanEquals<i32>({1, 2, 3}));
                CHECK_THAT(table.get(1, 0), SpanEquals<i32>({4, 5, 6}));
                CHECK_THAT(table.get(0, 1), SpanEquals<i32>({7, 8, 9}));
                CHECK_THAT(table.get(1, 1), SpanEquals<i32>({10, 11, 12}));
            }

            WHEN("Span select by value") {
                CHECK_THAT(table(-0.5, -0.5), SpanEquals<i32>({1, 2, 3}));
                CHECK_THAT(table(0.5, -0.5), SpanEquals<i32>({4, 5, 6}));
                CHECK_THAT(table(-0.5, 0.5), SpanEquals<i32>({7, 8, 9}));
                CHECK_THAT(table(0.5, 0.5), SpanEquals<i32>({10, 11, 12}));
            }
        }
    }
}

