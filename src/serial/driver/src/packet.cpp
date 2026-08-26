#include <rm_serial_driver/packet.hpp>

#include <algorithm>
#include <cstring>

namespace rm::serial
{
ReceivePacket FromVector(const std::vector<uint8_t>& data)
{
  ReceivePacket packet;
  std::memcpy(&packet, data.data(), std::min(data.size(), sizeof(packet)));
  return packet;
}

std::vector<uint8_t> ToVector(const SendPacket& data, bool include_arm_extension)
{
  std::vector<uint8_t> packet(SendPacketSize(include_arm_extension));
  std::memcpy(packet.data(), &data, packet.size());
  return packet;
}

std::size_t SendPacketSize(bool include_arm_extension)
{
  return include_arm_extension ? sizeof(SendPacket) : LegacySendPacketSize();
}

std::size_t LegacySendPacketSize()
{
  return sizeof(uint8_t) + sizeof(NavPacket) + sizeof(uint8_t);
}

bool HasValidArmReceiveExtension(const ReceivePacket& packet, std::size_t size)
{
  return size >= sizeof(ReceivePacket) &&
         packet.header == kReceiveHeader &&
         packet.end_frame == kLegacyEndFrame &&
         packet.arm_header == kArmExtensionHeader &&
         packet.arm_end_frame == kArmExtensionEndFrame &&
         packet.arm_state_valid != 0;
}
} 
