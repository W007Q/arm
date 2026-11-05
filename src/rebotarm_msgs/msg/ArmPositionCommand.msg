# Absolute arm positions in the canonical joint1..joint6 order.
# The STM32 firmware interprets velocity_limits as non-negative position
# trajectory limits; this is not a signed target velocity.
builtin_interfaces/Time stamp
float64[6] positions
float64[6] velocity_limits
